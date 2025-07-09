import os
pretrain_path = '/data/qingsong/pretrain'

from models_mae import mae_vit_base_patch16
import torch
vit = mae_vit_base_patch16()
checkpoint = torch.load(os.path.join(pretrain_path, 'mae_pretrain_vit_base_full.pth'), map_location="cpu")['model']
vit.load_state_dict(checkpoint, strict=True)

import argparse
args = argparse.Namespace(
    num_layers=12,
    vocab_size=1,
    hidden_size=768,
    num_attention_heads=12,
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
    dec_num_layers=8,
    dec_hidden_size=512,
    dec_num_attention_heads=16,
    load=None
    )