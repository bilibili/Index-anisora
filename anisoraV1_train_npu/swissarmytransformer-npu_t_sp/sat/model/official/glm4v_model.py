from .eva_clip_model import EVA2CLIPModel
from .chatglm4_model import ChatGLM4Model

import json
import os
import torch
import torch.nn.functional as F
from sat.model.base_model import BaseMixin
import math
import torch.nn as nn
from sat import mpu
from sat.helpers import print_rank0
import torch.nn.init as init
from sat.training.model_io import extract_model_specific_args_to_dump
import argparse
from copy import deepcopy

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class GLU(nn.Module):
    def __init__(self, args, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, args.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(args.inner_hidden_size, args.hidden_size, bias=False)
        # self.norm2 = nn.LayerNorm(args.hidden_size)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        # x = self.norm2(x)
        return x

def override_dist_dtype_device_args(args, b={}):
    if args.mode == 'inference':
        minimal_args = argparse.Namespace(
            world_size=args.world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            skip_init=args.skip_init,
            use_gpu_initialization=args.use_gpu_initialization,
            deepspeed=args.deepspeed,
            bf16=args.bf16,
            fp16=args.fp16,
            mode=args.mode,
            device=args.device
        )
    else:
        minimal_args = argparse.Namespace(
                world_size=args.world_size,
                rank=args.rank,
                local_rank=args.local_rank,
                skip_init=args.skip_init,
                use_gpu_initialization=args.use_gpu_initialization,
                deepspeed=args.deepspeed,
                bf16=args.bf16,
                fp16=args.fp16,
                mode=args.mode,
                checkpoint_activations=args.checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                device=args.device,
                hidden_dropout=0.,
                attention_dropout=0.
            )
    if hasattr(args, 'model_parallel_size'):
        b['model_parallel_size'] = args.model_parallel_size
    return argparse.Namespace(**deepcopy(b), **vars(minimal_args))

class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        # Option 1. if not loading from ckpt, using this code
        if args.eva_args:
            vit_args = override_dist_dtype_device_args(args, args.eva_args)
            self.vit_model = EVA2CLIPModel(EVA2CLIPModel.get_args(**vars(vit_args)))
        # ===============================================
        # Option 2. if loading from vit checkpoint, use this code
        else:
            url = os.path.join(os.getenv("SAT_HOME"), 'eva-clip-4b-14-x-drop-last-layer')
            print("loading vit checkpoint from", url)
            vit_args = override_dist_dtype_device_args(args, args.eva_args)
            self.vit_model, vit_args = EVA2CLIPModel.from_pretrained(url, vit_args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size > 1 else {})
            args.eva_args = extract_model_specific_args_to_dump(vit_args, self.vit_model)
            print("loading finished", url)
        
        args.proj_hidden_size = args.hidden_size if args.proj_hidden_size is None else args.proj_hidden_size
        self.conv = nn.Conv2d(in_channels=self.vit_model.transformer.hidden_size, out_channels=args.proj_hidden_size, kernel_size=2, stride=2)
        self.linear_proj = GLU(args, args.proj_hidden_size)
        self.linear_proj.apply(init_weights)

        self.image_length = args.image_length
        self.boi = nn.Parameter(torch.ones(1, 1, args.hidden_size).float())
        self.eoi = nn.Parameter(torch.ones(1, 1, args.hidden_size).float())
        init.xavier_uniform_(self.boi)
        init.xavier_uniform_(self.eoi)

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        vision_inputs = {}
        for k in kw_args:
            if k.startswith('vision_') and k != 'vision_expert_mask':
                vision_inputs[k[7:]] = kw_args[k]
        if input_ids.shape[1] == 1 or not vision_inputs:
            word_embedding = self.transformer.word_embeddings(input_ids)
        else:
            if 'position_ids' not in vision_inputs:
                vision_inputs['position_ids'] = None
            image_emb = self.vit_model(**vision_inputs)[0]
            b, s, e = image_emb.shape # (b, 6400, 1792)
            grid_size = int(s**0.5)
            image_emb = image_emb.view(b, grid_size, grid_size, e).permute(0,3,1,2) # (b, 1792, 80, 80)
            image_emb = self.conv(image_emb) # (b, 4096, 40, 40)
            image_emb = image_emb.flatten(2).transpose(1, 2) # (b, 1600, 4096)
            image_emb = self.linear_proj(image_emb) # (b, 1600, 6656)

            image_embed_mask = kw_args['image_embed_mask']
            
            word_embedding = self.transformer.word_embeddings(input_ids).clone()
            word_embedding[image_embed_mask.bool()] = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1).reshape(-1, image_emb.shape[-1])
            word_embedding = word_embedding.contiguous()

        return word_embedding

class GLM4VModel(ChatGLM4Model):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GLM4V', 'GLM4V Configurations')
        group.add_argument('--image_length', type=int, default=256)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--proj_hidden_size', type=int, default=None)
        return super().add_model_specific_args(parser)

    def forward(self, input_ids, **kwargs):
        if input_ids.shape[1] > 1:
            return super().forward(input_ids=input_ids, **kwargs)
        if "vision_expert_mask" in kwargs:
            kwargs.pop("vision_expert_mask")
        if "image_embed_mask" in kwargs:
            kwargs.pop("image_embed_mask")
        return super().forward(input_ids=input_ids, **kwargs)