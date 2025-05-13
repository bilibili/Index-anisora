import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel, non_conflict
from sat.model.official.vit_model import ViTModel, ClsMixin
from sat.model.mixins import BaseMixin
from sat import mpu

class AttnMixin(BaseMixin):
    def __init__(self, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.proj_l = nn.ModuleList([nn.Linear(num_heads, num_heads) for i in range(num_layers)])
        self.proj_w = nn.ModuleList([nn.Linear(num_heads, num_heads) for i in range(num_layers)])

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        # adapted from https://github.com/THUDM/sat/blob/main/sat/mpu/transformer.py#L47
        if scaling_attention_score:
            query_layer = query_layer * (query_layer.shape[-1]**-0.5) # / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if log_attention_weights is not None:
            attention_scores += log_attention_weights
        
        attention_scores = self.proj_l[kwargs['layer_id']](attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.mul(attention_scores, attention_mask) - \
                            10000.0 * (1.0 - attention_mask)

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = self.proj_w[kwargs['layer_id']](attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    def reinit(self, parent_model=None):
        # init with identity matrix so that pretrained weights with standard_attn can be reused
        for i in range(self.num_layers):
            nn.init.eye_(self.proj_l[i].weight)
            nn.init.eye_(self.proj_w[i].weight)

class EncForward(BaseMixin):
    def __init__(self, dim, num_layers, init_values=1e-4):
        super().__init__()
        self.gamma_1 = nn.ParameterList([nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) for i in range(num_layers)])
        self.gamma_2 = nn.ParameterList([nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) for i in range(num_layers)])
    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        layer = self.transformer.layers[kw_args['layer_id']]

        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(layernorm_output1, mask, **kw_args)

        # Residual connection.
        layernorm_input = hidden_states + self.gamma_1[kw_args['layer_id']] * attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)

        # Second residual connection.
        output = layernorm_input + self.gamma_2[kw_args['layer_id']] * mlp_output

        return output

from sat.model.transformer import standard_attention
from sat.mpu.utils import split_tensor_along_last_dim

class DecForward(BaseMixin):
    def __init__(self, dim, num_layers, init_values=1e-4):
        super().__init__()
        self.gamma_1 = nn.ParameterList([nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) for i in range(num_layers)])
        self.gamma_2 = nn.ParameterList([nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) for i in range(num_layers)])
    
    def position_embedding_forward(self, position_ids, **kwargs):
        return 0

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        encoder_outputs = kw_args['encoder_outputs']
        assert encoder_outputs is not None
        # Layer norm at the begining of the transformer layer.
        u = torch.cat([hidden_states, encoder_outputs], 1)
        layernorm_output1 = layer.input_layernorm(u)
        assert 'cross_attention_mask' in kw_args
        # Cross attention
        attention_output = layer.cross_attention(layernorm_output1, **kw_args)
        # Residual connection.
        layernorm_input = hidden_states + self.gamma_1[kw_args['layer_id']] * attention_output
        # Layer norm post the cross attention
        layernorm_output = layer.post_cross_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)

        # Second residual connection.
        output = layernorm_input + self.gamma_2[kw_args['layer_id']] * mlp_output

        return output

    def cross_attention_forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # adapted from https://github.com/THUDM/sat/blob/d8c9d1e0a9bb2af1e1d26a68b35f16d84aafcc2f/sat/mpu/transformer.py#L216
        # if you want to use a customized attention_fn, just inherit the attention mixin for this mixin and use self.attention_fn instead of self.hooks['attention_fn']
        layer = self.transformer.layers[kw_args['layer_id']].cross_attention
        
        attention_fn = standard_attention
        
        mixed_query_layer = layer.query(hidden_states[:, :1])
        mixed_x_layer = layer.key_value(hidden_states)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)

        dropout_fn = layer.attention_dropout if layer.training else None
        # Reshape and transpose [b, np, s, hn]
        query_layer = layer._transpose_for_scores(mixed_query_layer)
        key_layer = layer._transpose_for_scores(mixed_key_layer)
        value_layer = layer._transpose_for_scores(mixed_value_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn,
                                        cross_attention=True, **kw_args)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (layer.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = layer.dense(context_layer)
        if layer.training:
            output = layer.output_dropout(output)

        return output

class CaiTEncoder(ViTModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6, use_final_layernorm=False):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, use_final_layernorm=use_final_layernorm)
        self.del_mixin('cls')
        self.add_mixin('attn', AttnMixin(args.num_attention_heads, args.num_layers))
        self.add_mixin('enc_forward', EncForward(args.hidden_size, args.num_layers, init_values=args.init_scale))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CaiT-enc', 'CaiT encoder Configurations')
        group.add_argument('--init-scale', type=float, default=1e-4)
        return super().add_model_specific_args(parser)

class CaiTDecoder(BaseModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        super().__init__(args, is_decoder=True, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('cls', ClsMixin(args.hidden_size, args.num_classes))
        self.add_mixin('dec_forward', DecForward(args.hidden_size, args.num_layers, init_values=args.init_scale))
    @classmethod
    def add_model_specific_args(cls, parser):
        return super().add_model_specific_args(parser)

from sat.model import EncoderDecoderModel
import argparse

class CaiT(EncoderDecoderModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        encoder = CaiTEncoder(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        dec_args = argparse.Namespace(**vars(args))
        # dec_args.enc_hidden_size = dec_args.hidden_size  # used for cross attn
        override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_order',
                            'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            dec_attr = getattr(dec_args, 'dec_' + name, None)
            if dec_attr is not None:  # else use encoder-config
                setattr(dec_args, name, dec_attr)
        decoder = CaiTDecoder(dec_args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        super().__init__(args, encoder=encoder, decoder=decoder)
        
    def forward(self, input_ids, enc_position_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        # Please use self.decoder for auto-regressive generation.
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
        return (encoder_outputs, decoder_outputs, *mems)
