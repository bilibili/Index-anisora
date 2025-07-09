import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel, non_conflict
from sat.model.official.vit_model import ViTModel
from sat.model.mixins import BaseMixin
from sat import mpu

from sat.model.position_embedding import get_2d_sincos_pos_embed

"""
MAE model follows encoder-decoder architecture.
For encoder, it is a normal ViTModel with customed position embeddings.
For decoder, it is a normal BaseModel adding [MASK] token.
"""

from sat.model.official.vit_model import InterpolatedPositionEmbeddingMixin

class PosMixin(InterpolatedPositionEmbeddingMixin):
    def __init__(self, hidden_size, old_property, property, init_method_std=0.02):
        super().__init__(hidden_size, old_property, property, init_method_std=init_method_std)
        self.hidden_size = hidden_size

    def reinit(self, parent_model=None):
        old_weight = self.transformer.position_embeddings.weight.data
        self.transformer.position_embeddings = torch.nn.Embedding(self.property.seq_len, old_weight.shape[1]).type(old_weight.dtype).to(old_weight.device).requires_grad_(False)
        self.transformer.position_embeddings.weight.data = torch.Tensor(get_2d_sincos_pos_embed(self.hidden_size, self.property.grid_size, self.property.pre_len, self.property.post_len))
    
    def after_position_forward(self, hidden_states, **kw_args):
        """
        Perform random_masking after adding position_embedding.
        """
        x = hidden_states[:, 1:]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, kw_args['mask_ratio'])

        # append cls token
        cls_tokens = hidden_states[:, :1]
        x = torch.cat((cls_tokens, x), dim=1)
        return x, {'mask': mask, 'ids_restore': ids_restore}
    
    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        if kw_args['layer_id'] == 0:
            hidden_states, dic_buffer = self.after_position_forward(hidden_states, **kw_args)
            for k in dic_buffer:
                kw_args['output_this_layer'][k] = dic_buffer[k]
        output = layer(hidden_states, mask, *args, **kw_args)

        return output

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    

class MAEEncoder(ViTModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.del_mixin('cls')
        self.del_mixin('pos_embedding')
        self.add_mixin('pos_embedding', PosMixin(args.hidden_size, self.old_property, self.property))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('MAE-enc', 'MAE encoder Configurations')
        return super().add_model_specific_args(parser)


class MaskMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.decoder_embed = nn.Linear(args.enc_hidden_size, args.hidden_size, bias=True)
        self.decoder_pred = nn.Linear(args.hidden_size, args.patch_size**2 * args.in_channels, bias=True) # decoder to patch

    def word_embedding_forward(self, input_ids, **kwargs):
        x = kwargs["encoder_outputs"]
        ids_restore = kwargs["ids_restore"]

        x = self.decoder_embed(x)

        mask_tokens = self.transformer.word_embeddings(input_ids).repeat(1, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # breakpoint()
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        return x
    
    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return self.transformer.position_embeddings(position_ids)
    
    def final_forward(self, logits, **kw_args):
        logits = self.decoder_pred(logits)
        return logits[:, 1:]

class MAEDecoder(BaseModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('mask_forward', MaskMixin(args))
    @classmethod
    def add_model_specific_args(cls, parser):
        return super().add_model_specific_args(parser)

from sat.model import EncoderDecoderModel
import argparse

class MAE(EncoderDecoderModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        encoder = MAEEncoder(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        dec_args = argparse.Namespace(**vars(args))
        # dec_args.enc_hidden_size = dec_args.hidden_size  # used for cross attn
        override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads',
                            'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            dec_attr = getattr(dec_args, 'dec_' + name, None)
            if dec_attr is not None:  # else use encoder-config
                setattr(dec_args, name, dec_attr)
        setattr(dec_args, 'enc_hidden_size', args.hidden_size)
        decoder = MAEDecoder(dec_args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        super().__init__(args, encoder=encoder, decoder=decoder, tie_word_embeddings=False)
    
    def encode(self, input_ids, position_ids, attention_mask=None, **kw_args):
        return self.encoder(input_ids, position_ids, attention_mask, **kw_args)
    
    def decode(self, input_ids, position_ids, attention_mask, encoder_outputs, ids_restore, **kw_args):
        return self.decoder(input_ids, position_ids, attention_mask, encoder_outputs=encoder_outputs, ids_restore=ids_restore, **kw_args)

    def forward(self, input_ids, enc_position_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, **kw_args):
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=input_ids.device)
        encoder_outputs, *encoder_mems = self.encode(input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *decoder_mems = self.decode(input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, ids_restore=encoder_mems[0]["ids_restore"], **kw_args)
        return encoder_outputs, decoder_outputs, encoder_mems, decoder_mems

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder.property.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs