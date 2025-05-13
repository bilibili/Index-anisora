import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel, non_conflict
from sat.model.transformer import standard_attention
from sat import mpu
import math

class GPTNeoTypeMixin(BaseMixin):
    def __init__(self):
        super().__init__()
        
    def word_embedding_forward(self, input_ids, **kwargs):
        if "token_type_ids" in kwargs:
            return self.transformer.word_embeddings(input_ids) + self.transformer.word_embeddings(kwargs["token_type_ids"])
        else:
            return self.transformer.word_embeddings(input_ids)
        
class GPTNeoAttentionMixin(BaseMixin):
    def __init__(self, attention_types, window_size, max_positions):
        super().__init__()
        self.attention_types = attention_types
        self.window_size = window_size
        self.max_positions = max_positions
        self.bias_global = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
            1, 1, max_positions, max_positions
        )
        self.bias_local = torch.bitwise_xor(self.bias_global, torch.tril(self.bias_global, -window_size))
        
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                     attention_dropout=None, log_attention_weights=None, scaling_attention_score=False, **kwargs):
        
        attention_type = self.attention_types[kwargs['layer_id']]
        if attention_type not in ["global", "local"]:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `attention_type`: "
                f"{attention_type}. Select attn layer types from ['global', 'local'] only."
            )
        
        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        '''
        2022/08/02
        Difference to SAT-Base is the causal_mask.
        '''
        query_length, key_length = query_layer.size(-2), key_layer.size(-2)
        if attention_type == 'global':
            bias = self.bias_global
        else:
            bias = self.bias_local
        causal_mask = bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool).to(attention_scores.device)
        mask_value = torch.finfo(attention_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attention_scores.dtype).to(attention_scores.device)
        attention_scores = torch.where(causal_mask, attention_scores, mask_value)
        
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

        if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.where(attention_mask.to(attention_scores.device), attention_scores, mask_value)
            # attention_scores = torch.mul(attention_scores, attention_mask) - \
                            # 10000.0 * (1.0 - attention_mask)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

class GPTNeoModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer, **kwargs)
        self.add_mixin("gpt-type", GPTNeoTypeMixin())
        self.add_mixin("gpt-attn", GPTNeoAttentionMixin(args.attention_types, args.window_size, args.max_sequence_length))
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GPTNeo', 'GPTNeo Configurations')
        group.add_argument('--attention-types', type=str)
        group.add_argument('--window-size', type=str)
        return parser