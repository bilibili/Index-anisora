import os
pretrain_path = '/data/qingsong/pretrain'

from transformers import CLIPProcessor, CLIPModel
import torch
vit = CLIPModel.from_pretrained(os.path.join(pretrain_path, 'clip-vit-base-patch32'))
processor = CLIPProcessor.from_pretrained(os.path.join(pretrain_path, 'clip-vit-base-patch32'))

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
    patch_size=32,
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
    text_num_layers=12,
    text_hidden_size=512,
    text_num_attention_heads=8,
    text_vocab_size=49408,
    text_max_sequence_length=77,
    load=None,
    logit_scale_init_value=2.6592,
    projection_dim=512
    )