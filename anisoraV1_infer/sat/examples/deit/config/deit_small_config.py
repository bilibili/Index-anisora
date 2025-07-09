import os
pretrain_path = '/data/qingsong/pretrain'

import timm
import torch
vit = timm.create_model('deit_small_patch16_224', pretrained=False)
checkpoint = torch.load(os.path.join(pretrain_path, 'deit_small_patch16_224-cd65a155.pth'), map_location="cpu")
vit.load_state_dict(checkpoint["model"], strict=True)

import argparse
args = argparse.Namespace(
    num_layers=12,
    vocab_size=1,
    hidden_size=384,
    num_attention_heads=6,
    hidden_dropout=0.,
    attention_dropout=0.,
    in_channels=3,
    image_size=[224, 224],
    patch_size=16,
    pre_len=1,
    post_len=0,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=True,
    checkpoint_num_layers=1,
    sandwich_ln=False,
    post_ln=False,
    model_parallel_size=1,
    world_size=1,
    rank=0,
    num_classes=1000,
    load=None
    )