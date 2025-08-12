import os
pretrain_path = '/data/qingsong/pretrain'

import timm
import torch
vit = timm.create_model('cait_s24_224', pretrained=False)
checkpoint = torch.load(os.path.join(pretrain_path, 'S24_224.pth'), map_location="cpu")['model']
checkpoint_no_module = {}
for k, v in checkpoint.items():
    checkpoint_no_module[k.replace('module.', '')] = v
vit.load_state_dict(checkpoint_no_module, strict=True)

import argparse
args = argparse.Namespace(
    init_scale=1e-5,
    num_layers=24,
    vocab_size=1,
    hidden_size=384,
    num_attention_heads=8,
    hidden_dropout=0.,
    attention_dropout=0.,
    in_channels=3,
    image_size=[224, 224],
    patch_size=16,
    pre_len=0,
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
    dec_num_layers=2,
    load=None
    )