import os
import math
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel, non_conflict, load_checkpoint, get_model
from sat.model.official.vit_model import ViTModel, ImagePatchEmbeddingMixin
from sat.model.mixins import BaseMixin
from sat import mpu
from sat.model.transformer import LayerNorm
from sat import update_args_with_file
from sat.resources import auto_create

"""
CLIP model follows Siamese architecture.
For image encoder, it is a ViTModel with 32x32 patch.
For text encoder, it is a BaseModel with causal mask.
"""

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)

class ImageMixin(BaseMixin):
    def __init__(self, vision_embed_dim, projection_dim, layernorm_epsilon):
        super().__init__()
        self.pre_layernorm = LayerNorm(vision_embed_dim, eps=layernorm_epsilon)
        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
    
    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        if kw_args['layer_id'] == 0:
            hidden_states = self.pre_layernorm(hidden_states)
        output = layer(hidden_states, mask, *args, **kw_args)
        return output

    def final_forward(self, logits, **kw_args):
        return self.visual_projection(logits[:, 0])

class PatchMixin(ImagePatchEmbeddingMixin):
    def __init__(self, in_channels, hidden_size, property):
        super().__init__(in_channels, hidden_size, property)
        self.property = property
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=property.patch_size, stride=property.patch_size, bias=False)


class ImageEncoder(ViTModel):
    def __init__(self, args, layernorm_epsilon=1e-5, activation_func=QuickGELUActivation()):
        super().__init__(args, layernorm_epsilon=layernorm_epsilon, activation_func=activation_func)
        self.del_mixin('cls')
        self.add_mixin('image_enc', ImageMixin(args.hidden_size, args.projection_dim, layernorm_epsilon))
        self.del_mixin('patch_embedding')
        self.add_mixin('patch_embedding', PatchMixin(args.in_channels, args.hidden_size, self.property))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CLIP-image', 'CLIP image encoder Configurations')
        group.add_argument('--projection-dim', type=int)
        return super().add_model_specific_args(parser)

class TextMixin(BaseMixin):
    def __init__(self, text_embed_dim, projection_dim):
        super().__init__()
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)
    
    def final_forward(self, logits, **kw_args):
        return self.text_projection(logits[:, -1])

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        # causal mask
        mask = mask - mask.triu(1)
        layer = self.transformer.layers[kw_args['layer_id']]
        output = layer(hidden_states, mask, *args, **kw_args)
        return output

class TextEncoder(BaseModel):
    def __init__(self, args, layernorm_epsilon=1e-5, activation_func=QuickGELUActivation()):
        super().__init__(args, layernorm_epsilon=layernorm_epsilon, activation_func=activation_func)
        self.add_mixin('text_enc', TextMixin(args.hidden_size, args.projection_dim))

    # @classmethod
    # def add_model_specific_args(cls, parser):
    #     return super().add_model_specific_args(parser)

import argparse

class CLIP(nn.Module):
    def __init__(self, args, layernorm_epsilon=1e-5):
        super().__init__()
        self.image_encoder = ImageEncoder(args, layernorm_epsilon=layernorm_epsilon)
        text_args = argparse.Namespace(**vars(args))
        override_attrs = ['vocab_size', 'num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_order',
                            'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
        for name in override_attrs:
            text_attr = getattr(text_args, 'text_' + name, None)
            if text_attr is not None:  # else use encoder-config
                setattr(text_args, name, text_attr)
        self.text_encoder = TextEncoder(text_args, layernorm_epsilon=layernorm_epsilon)
        self.logit_scale = nn.Parameter(torch.ones([]) * args.logit_scale_init_value)
        
    def encode_image(self, input_ids, position_ids, attention_mask=None, **kw_args):
        return self.image_encoder(input_ids, position_ids, attention_mask, **kw_args)
    
    def encode_text(self, input_ids, position_ids, attention_mask, **kw_args):
        return self.text_encoder(input_ids, position_ids, attention_mask, **kw_args)
    
    def reinit(self, mixin_names): # please use different mixin names for two encoders
        self.image_encoder.reinit(mixin_names)
        self.text_encoder.reinit(mixin_names)
    
    def forward(self, image_input_ids, image_position_ids, text_input_ids, text_position_ids, *, image_attention_mask=None, text_attention_mask=None, **kw_args):
        image_embeds, *image_mems = self.encode_image(image_input_ids, image_position_ids, attention_mask=image_attention_mask, **kw_args)
        text_embeds, *text_mems = self.encode_text(text_input_ids, text_position_ids, attention_mask=text_attention_mask, **kw_args)
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        return image_embeds, text_embeds, logits_per_text, logits_per_image

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('SiameseModel', 'CLIP')
        group.add_argument("--text-layernorm-order", type=str, default=None)
        group.add_argument("--text-num-layers", type=int, default=None)
        group.add_argument("--text-hidden-size", type=int, default=None)
        group.add_argument("--text-num-attention-heads", type=int, default=None)
        group.add_argument("--text-max-sequence-length", type=int, default=None)
        group.add_argument("--text-inner-hidden-size", type=int, default=None)
        group.add_argument("--text-hidden-size-per-attention-head", type=int, default=None)
        group.add_argument("--logit-scale-init-value", type=float, default=None)
        return parser

    @classmethod
    def from_pretrained(cls, args, name, *, path=None, url=None):
        model_path = auto_create(name, path=path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(args, cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args
