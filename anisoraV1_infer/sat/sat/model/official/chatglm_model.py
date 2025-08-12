import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
import math
from sat import mpu
from sat.mpu.utils import split_tensor_along_last_dim
from sat.transformer_defaults import attention_fn_default

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))

def gelu(x):
    return gelu_impl(x)

from sat.model.position_embedding.rotary_embeddings import RotaryEmbedding, apply_rotary_pos_emb_index
from sat.mpu.layers import ColumnParallelLinear

class ChatGLMFinalMixin(BaseMixin):
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

class ChatGLMAttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            hidden_size // (num_heads * 2),
            base=10000,
            precision=torch.half,
            learnable=False,
        )

    def attention_forward(self, hidden_states, mask, **kw_args):
        mixin_self = self
        position_ids = kw_args['position_ids']
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
        
        query_layer = query_layer.permute(2, 0, 1, 3)
        key_layer = key_layer.permute(2, 0, 1, 3)
        value_layer = value_layer.permute(2, 0, 1, 3)

        q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
        k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
        cos, sin = mixin_self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
            position_ids[:, 1, :].transpose(0, 1).contiguous()
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        query_layer = torch.cat([q1, q2], dim=(q1.ndim - 1))
        key_layer = torch.cat([k1, k2], dim=(k1.ndim - 1))

        if kw_args.get('past_key_values', None) is not None:
            pack = kw_args['past_key_values'][kw_args['layer_id']]
            if pack is not None:
                past_key, past_value = pack
                key_layer = torch.cat((past_key, key_layer), dim=0)
                value_layer = torch.cat((past_value, value_layer), dim=0)
        kw_args['output_this_layer']['past_key_values'] = (key_layer, value_layer)

        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        dropout_fn = self.attention_dropout if self.training else None

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class ChatGLMLayerMixin(BaseMixin):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        num_layers = self.num_layers
        self = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        attention_input = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(attention_input, mask, **kw_args)
        alpha = (2 * num_layers) ** 0.5

        # Third LayerNorm
        if self.layernorm_order == 'sandwich':
            attention_output = self.third_layernorm(attention_output)
        
        # Residual connection.
        if self.layernorm_order == 'post':
            hidden_states = attention_input * alpha + attention_output
        else:
            hidden_states = hidden_states + attention_output

        
        mlp_input = self.post_attention_layernorm(hidden_states)

        if self.is_decoder:
            encoder_outputs = kw_args['encoder_outputs']
            if encoder_outputs is not None:
                assert 'cross_attention_mask' in kw_args
                # Cross attention
                attention_output = self.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = mlp_input + attention_output
                # Layer norm post the cross attention
                mlp_input = self.post_cross_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input, **kw_args)

        # Fourth LayerNorm
        if self.layernorm_order == 'sandwich':
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        if self.layernorm_order == 'post':
            output = mlp_input * alpha + mlp_output
        else:
            output = hidden_states + mlp_output

        return output
    
class ChatGLMModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLMModel, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin("chatglm-final", ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("chatglm-attn", ChatGLMAttnMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin("chatglm-layer", ChatGLMLayerMixin(args.num_layers))
        self.bos_token_id = args.bos_token_id
        self.mask_token_id = args.mask_token_id
        self.gmask_token_id = args.gmask_token_id
        self.pad_token_id = args.pad_token_id

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None
    
    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if attention_mask is None and position_ids is None:
            attention_mask, position_ids = self.get_inputs(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        if attention_mask is not None and attention_mask.dtype is torch.bool:
            attention_mask = (~attention_mask).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    
    def get_inputs(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        if attention_mask is None:
            if past_key_values is not None and input_ids.size(0) == 1:
                attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
            else:
                attention_mask = self.get_masks(
                    input_ids=input_ids,
                    device=input_ids.device, **kwargs
                )
        if position_ids is None:
            MASK, gMASK = self.mask_token_id, self.gmask_token_id
            mask_token = gMASK if gMASK in input_ids else MASK
            use_gmask = True if gMASK in input_ids else False

            mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
            position_ids = self.get_position_ids(
                input_ids=input_ids,
                mask_positions=mask_positions,
                device=input_ids.device,
                gmask=use_gmask, **kwargs
            )
        return attention_mask, position_ids
    
    def get_pad_length(self, seq):
        l = 0
        while l < len(seq) and seq[l] == self.pad_token_id:
            l += 1
        return l
    
    def get_masks(self, input_ids, device, **kwargs):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), dtype=next(self.parameters()).dtype, device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        for i, pad_length in enumerate(pad_lengths):
            attention_mask[i, :, :pad_length] = 0
            attention_mask[i, :pad_length, :] = 0
        attention_mask.unsqueeze_(1)
        # attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, gmask=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        position_ids = [torch.arange(seq_length-pad_length, dtype=torch.long, device=device) for pad_length in pad_lengths]
        for i, (context_length, pad_length) in enumerate(zip(context_lengths, pad_lengths)):
            position_ids[i][context_length-pad_length:] = mask_positions[i] - pad_length
        block_position_ids = [torch.cat((
            torch.zeros(context_length, dtype=torch.long, device=device),
            torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        )) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0)
        position_ids = [torch.cat((
            torch.zeros(pad_length, dtype=torch.long, device=device),
            range_pos
        )) for pad_length, range_pos in zip(pad_lengths, position_ids)]
        position_ids = torch.stack(position_ids, dim=0)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        return position_ids
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM', 'ChatGLM Configurations')
        group.add_argument('--bos-token-id', type=int)
        group.add_argument('--mask-token-id', type=int)
        group.add_argument('--gmask-token-id', type=int)
        group.add_argument('--pad-token-id', type=int)
        return super().add_model_specific_args(parser)
