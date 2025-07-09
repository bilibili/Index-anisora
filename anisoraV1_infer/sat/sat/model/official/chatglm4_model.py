import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
from sat.mpu.utils import split_tensor_along_last_dim

from sat.ops.layernorm import RMSNorm
from sat.transformer_defaults import attention_fn_default
from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding
from sat.mpu.layers import ColumnParallelLinear

class ChatGLM4AttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads, max_seq_len):
        super().__init__()
        rotary_dim = hidden_size // num_heads
        self.rotary_pos_emb = FastRotaryEmbedding(rotary_dim // 2)
        self.max_seq_len = max_seq_len
        
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

        max_seq_len = kw_args['position_ids'].max() + 1
        query_layer, key_layer = origin.rotary_pos_emb(query_layer, key_layer, kw_args['position_ids'], max_seqlen=max_seq_len)

        if kw_args.get('past_key_values', None) is not None:
            pack = kw_args['past_key_values'][kw_args['layer_id']]
            if pack is not None:
                past_key, past_value = pack
                key_layer = torch.cat((past_key, key_layer), dim=2)
                value_layer = torch.cat((past_value, value_layer), dim=2)
        kw_args['output_this_layer']['past_key_values'] = (key_layer, value_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList([ColumnParallelLinear(
            in_features,
            hidden_features,
            gather_output=False,
            # init_method=init_method,
            bias=bias,
            # params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h_gate",
            # skip_init=skip_init,
            # device=device
        ) for i in range(num_layers)])

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args['layer_id']].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args['layer_id']](x)
        hidden = origin.activation_func(x1) * x2
        x = origin.dense_4h_to_h(hidden)
        return x

from .chatglm_model import ChatGLMFinalMixin

class ChatGLM4Model(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLM4Model, self).__init__(args, transformer=transformer, activation_func=F.silu, layernorm=RMSNorm, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin("chatglm-final", ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("attn", ChatGLM4AttnMixin(args.hidden_size, args.num_attention_heads, args.max_sequence_length))
        if not (hasattr(args, 'is_gated_mlp') and args.is_gated_mlp):
            self.add_mixin("mlp", SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, bias=args.use_bias))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None
    
    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, dtype=next(self.parameters()).dtype, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length, dtype=next(self.parameters()).dtype,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask
    
    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids
    
    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if attention_mask is not None and attention_mask.ndim == 4:
            pass
        elif past_key_values is not None and input_ids.size(0) == 1:
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = ~attention_mask
        attention_mask = attention_mask.to(next(self.parameters()).dtype)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM4', 'ChatGLM4 Configurations')
        return super().add_model_specific_args(parser)