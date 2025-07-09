import os
pretrain_path = '/data/qingsong/pretrain'

import torch
from models.detector import Detector
yolos = Detector(
        num_classes=91,
        pre_trained=None,
        det_token_num=100,
        backbone_name='tiny',
        init_pe_size=[800, 1333],
        mid_pe_size=None,
        use_checkpoint=False,
    )
yolos.load_state_dict(torch.load(os.path.join(pretrain_path, 'yolos_ti.pth'))['model'], strict=False)

import argparse
args = argparse.Namespace(
    num_layers=12,
    vocab_size=101,
    num_det_tokens=100,
    hidden_size=192,
    num_attention_heads=3,
    hidden_dropout=0.,
    attention_dropout=0.,
    in_channels=3,
    image_size=[800, 1333],
    patch_size=16,
    pre_len=1,
    post_len=100,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=False,
    checkpoint_num_layers=1,
    sandwich_ln=False,
    post_ln=False,
    model_parallel_size=1,
    world_size=1,
    rank=0,
    load=None,
    num_classes=1000,
    mode='inference',
    num_det_classes=92
    )