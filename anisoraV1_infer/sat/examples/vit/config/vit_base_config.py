import os
pretrain_path = './'

import timm
vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=False)
vit.load_pretrained(os.path.join(pretrain_path, 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'))

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
    layernorm_epsilon=1e-6,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=None,
    layernorm_order='pre',
    skip_init=True,
    model_parallel_size=1,
    num_classes=21843,
    tokenizer_type='fake',
    mode='inference',
    save='vit-base-patch16-224-in21k'
    )