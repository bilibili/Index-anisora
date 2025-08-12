import torch
from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
import torch.nn as nn
from .vit_model import ViTProperty
from sat.ops.layernorm import LayerNorm

class MaskedPatchEmbedMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, property):
        super(MaskedPatchEmbedMixin, self).__init__()
        self.property = property
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=property.patch_size, stride=property.patch_size)

    def word_embedding_forward(self, input_ids, **kwargs):
        """
        Input:
        * input_ids with shape (batch_size, pre_len+post_len)
        * kwargs["image"] with shape (B, C, H, W)
        * kwargs["bool_masked_pos"] with shape (B, num_patches)
        Output:
        * (batch_size, pre_len+num_patches+post_len, hidden_size)
        """
        images = kwargs["image"]
        embeddings = self.proj(images)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        if kwargs.get("bool_masked_pos", None) is not None:
            batch_size, seq_len, _ = embeddings.size()
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = kwargs["bool_masked_pos"].unsqueeze(-1).type_as(mask_token)
            embeddings = embeddings * (1 - w) + mask_token * w
        pre_word_embeddings = self.transformer.word_embeddings(input_ids[:,:self.property.pre_len])
        post_word_embeddings = self.transformer.word_embeddings(input_ids[:,self.property.pre_len:self.property.pre_len+self.property.post_len])
        embeddings = torch.cat([pre_word_embeddings, embeddings, post_word_embeddings], dim=1)
        return embeddings

class EVA2FinalMixin(BaseMixin):
    def __init__(self, predict_feature_dim, hidden_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, predict_feature_dim)

    def final_forward(self, logits, **kwargs):
        logits = logits[:, 1:]
        if kwargs.get("bool_masked_pos", None) is not None:
            return self.lm_head(logits[kwargs["bool_masked_pos"]])
        return self.lm_head(logits)

class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, act_layer=nn.SiLU, drop=0., eps=1e-6):
        super().__init__()

        # self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.ModuleList([nn.Linear(in_features, hidden_features) for i in range(num_layers)])

        self.act = act_layer()
        self.ffn_ln = nn.ModuleList([LayerNorm(hidden_features, eps=eps) for i in range(num_layers)])
        # self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args['layer_id']].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args['layer_id']](x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln[kw_args['layer_id']](hidden)
        x = origin.dense_4h_to_h(x)
        x = self.drop(x)
        return x

# I don't know why the original eva2 model doesn't add bias to key, but adds bias to query and value. I just add bias to all of them here.
from sat.model.position_embedding.vision_rotary_embeddings import VisionRotaryEmbeddingFast
from sat.transformer_defaults import standard_attention
from sat.mpu.utils import split_tensor_along_last_dim
class EVA2AttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_attention_heads, property):
        super().__init__()
        half_head_dim = hidden_size // num_attention_heads // 2
        hw_seq_len = property.image_size[0] // property.patch_size
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )
    
    def attention_forward(self, hidden_states, mask, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = standard_attention
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)


        q_t = query_layer[:, :, 1:, :]
        ro_q_t = origin.rope(q_t)
        q = torch.cat((query_layer[:, :, :1, :], ro_q_t), -2).type_as(value_layer)

        k_t = key_layer[:, :, 1:, :]
        ro_k_t = origin.rope(k_t)
        k = torch.cat((key_layer[:, :, :1, :], ro_k_t), -2).type_as(value_layer)

        context_layer = attention_fn(q, k, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output


class EVA2Model(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        self.property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = self.property.seq_len
        super().__init__(args, transformer=transformer, **kwargs)
        self.add_mixin("patch_embedding", MaskedPatchEmbedMixin(args.in_channels, args.hidden_size, self.property))
        # The old_property of ViTModel is not elegent. However, I don't have time to fix them (including vit, cait, deit, yolos). I can only discard it since eva model for now.
        # self.add_mixin("pos_embedding", InterpolatedPositionEmbeddingMixin(args.hidden_size, self.old_property, self.property))
        self.add_mixin("eva2-final", EVA2FinalMixin(args.predict_feature_dim, args.hidden_size))
        self.add_mixin("eva2-mlp", SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, eps=args.layernorm_epsilon))
        self.add_mixin("eva2-attn", EVA2AttnMixin(args.hidden_size, args.num_attention_heads, self.property))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return self.transformer.position_embeddings.weight.unsqueeze(0)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EVA2', 'EVA2 Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1) # [cls] by default
        group.add_argument('--post-len', type=int, default=0) # empty by default, but sometimes with special tokens, such as [det] in yolos.
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--patch-size', type=int, default=14)
        group.add_argument('--predict-feature-dim', type=int, default=768)
        return parser


