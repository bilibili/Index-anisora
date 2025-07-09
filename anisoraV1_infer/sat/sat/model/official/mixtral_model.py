from sat.model import BaseMixin, BaseModel
import torch
import torch.nn as nn

from sat.transformer_defaults import attention_fn_default
from sat.mpu.utils import split_tensor_along_last_dim
import torch.nn.functional as F
from sat.mpu import ColumnParallelLinear

from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding

class RotaryMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.rotary_emb = FastRotaryEmbedding(hidden_size // num_heads, base=1000000)

    def attention_forward(self, hidden_states, mask, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = attention_fn_default
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        query_layer, key_layer = origin.rotary_emb(query_layer, key_layer, kw_args['position_ids'], max_seqlen=kw_args['position_ids'].max()+1, layer_id=kw_args['layer_id'])

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class MixtralMlpMixin(BaseMixin):
    def __init__(self, num_layers, in_features, num_experts, num_experts_per_tok):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.gates = nn.ModuleList([nn.Linear(in_features, num_experts, bias=False) for i in range(num_layers)])

    def routing_forward(self, hidden_states, **kw_args):
        # Adapted from mixtral-8x7b https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gates[kw_args['layer_id']](hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts

class LMMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = ColumnParallelLinear(
            hidden_size,
            vocab_size,
            gather_output=True,
            # init_method=init_method,
            bias=False,
            # params_dtype=params_dtype,
            module=self,
            name="lm_head",
            # skip_init=skip_init,
            # device=device
        )

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

from sat.ops.layernorm import RMSNorm

class MixtralModel(BaseModel):
    def __init__(self, args, transformer=None, layernorm=RMSNorm, activation_func=nn.functional.silu, **kwargs):
        super().__init__(args, transformer=transformer, layernorm=layernorm, activation_func=activation_func, init_method_std=0.01, **kwargs)
        del self.transformer.position_embeddings
        if 'inner_hidden_size' not in args:
            args.inner_hidden_size = None
        self.add_mixin("rotary", RotaryMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin("lm", LMMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("mlp", MixtralMlpMixin(args.num_layers, args.hidden_size, args.num_experts, args.num_experts_per_tok))
    
    def position_embedding_forward(self, *args, **kwargs):
        return None
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Mixtral-8x7b', 'Mixtral-8x7b Configurations')
        group.add_argument('--bos-token-id', type=int, default=1)
        group.add_argument('--eos-token-id', type=int, default=2)
        group.add_argument('--num-experts-per-tok', type=int, default=2)
        return parser
    
