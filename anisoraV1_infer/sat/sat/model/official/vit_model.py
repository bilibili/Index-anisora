# -*- encoding: utf-8 -*-
# @File    :   vit_model.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
# @Modified:   2022/04/08
# @By      :   Qingsong Lv
# @Contact :   lqs19@mails.tsinghua.edu.cn
import argparse

import torch

from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
import torch.nn as nn
from collections.abc import Iterable
import math
import torch.nn.functional as F


gelu = nn.functional.gelu


class ViTProperty:
    """
    Store some hyper-parameters such as image size and patch size.
    seq_len = pre_len + image_len + post_len
    """
    def __init__(self, image_size, patch_size, pre_len, post_len, **kwargs):
        assert isinstance(image_size, Iterable) and len(image_size) == 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.pre_len = pre_len
        self.post_len = post_len
        self.seq_len = self.pre_len + self.num_patches + self.post_len


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, property):
        super(ImagePatchEmbeddingMixin, self).__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=property.patch_size, stride=property.patch_size)

    def word_embedding_forward(self, input_ids, **kwargs):
        """
        Input:
        * input_ids with shape (batch_size, pre_len+post_len)
        * kwargs["image"] with shape (B, C, H, W)
        Output:
        * (batch_size, hidden_size)
        """
        images = kwargs["image"]
        embeddings = self.proj(images)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        pre_word_embeddings = self.transformer.word_embeddings(input_ids[:,:self.transformer.property.pre_len])
        post_word_embeddings = self.transformer.word_embeddings(input_ids[:,self.transformer.property.pre_len:self.transformer.property.pre_len+self.transformer.property.post_len])
        embeddings = torch.cat([pre_word_embeddings, embeddings, post_word_embeddings], dim=1)
        return embeddings


class InterpolatedPositionEmbeddingMixin(BaseMixin):
    def position_embedding_forward(self, position_ids, **kwargs):
        """
        There are two modes for position_embedding:
        * offline mode: You have reinited position_embeddings to a pre-defined new seq_len.
        * online mode: You need to interpolate position_embeddings for every forward pass.

        Input:
        * position_ids: (batch_size, seq_len)
        * kwargs["offline"]: boolean to identify offline or not
        * kwargs["height"], kwargs["width"]: specified image height and width for online mode
        """
        return self.transformer.position_embeddings.weight.unsqueeze(0)

    def reinit(self, parent_model=None, property=None):
        """
        new pre_len, new num_patches and new post_len should all be larger or equal than the old ones.
        """
        assert property is not None
        old_weight = self.transformer.position_embeddings.weight.data
        pre_weight = old_weight[:self.transformer.property.pre_len]
        post_weight = old_weight[self.transformer.property.pre_len+self.transformer.property.num_patches:]
        image_weight = old_weight[self.transformer.property.pre_len:self.transformer.property.pre_len+self.transformer.property.num_patches].reshape(1, self.transformer.property.grid_size[0], self.transformer.property.grid_size[1], -1).permute(0, 3, 1, 2)
        original_dtype = image_weight.dtype
        image_weight = F.interpolate(image_weight.float(), size=property.grid_size, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(property.num_patches, -1).to(original_dtype)
        self.transformer.position_embeddings = torch.nn.Embedding(property.seq_len, old_weight.shape[1]).type(old_weight.dtype).to(old_weight.device)
        torch.nn.init.normal_(self.transformer.position_embeddings.weight, mean=0.0, std=0.02)
        self.transformer.position_embeddings.weight.data[:self.transformer.property.pre_len] = pre_weight
        self.transformer.position_embeddings.weight.data[property.pre_len:property.pre_len+property.num_patches] = image_weight
        self.transformer.position_embeddings.weight.data[property.pre_len+property.num_patches:property.pre_len+property.num_patches+self.transformer.property.post_len] = post_weight
        self.transformer.property = property


class ClsMixin(BaseMixin):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def final_forward(self, logits, **kw_args):
        logits = self.classifier(logits[:, 0])
        return logits


class ViTModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = property.pre_len + property.num_patches + property.post_len
        if 'activation_func' not in kwargs:
            kwargs['activation_func'] = gelu
        super().__init__(args, transformer=transformer, **kwargs)
        self.transformer.property = property
        self.add_mixin("patch_embedding", ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, property))
        self.add_mixin("pos_embedding", InterpolatedPositionEmbeddingMixin())
        self.add_mixin("cls", ClsMixin(args.hidden_size, args.num_classes))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT', 'ViT Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1) # [cls] by default
        group.add_argument('--post-len', type=int, default=0) # empty by default, but sometimes with special tokens, such as [det] in yolos.
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--num-classes', type=int, default=21843)
        group.add_argument('--patch-size', type=int, default=16)
        return parser


