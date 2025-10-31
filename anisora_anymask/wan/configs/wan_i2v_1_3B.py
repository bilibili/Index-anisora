# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V 14B ------------------------#

i2v_1_3B = EasyDict(__name__='Config: Wan I2V 1_3B')
i2v_1_3B.update(wan_shared_cfg)

i2v_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_1_3B.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_1_3B.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_1_3B.clip_dtype = torch.float16
i2v_1_3B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
i2v_1_3B.clip_tokenizer = 'xlm-roberta-large'

# vae
i2v_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_1_3B.vae_stride = (4, 8, 8)

# transformer
i2v_1_3B.patch_size = (1, 2, 2)
i2v_1_3B.dim = 1536
i2v_1_3B.ffn_dim = 8960
i2v_1_3B.freq_dim = 256
i2v_1_3B.num_heads = 12
i2v_1_3B.num_layers = 30
i2v_1_3B.window_size = (-1, -1)
i2v_1_3B.qk_norm = True
i2v_1_3B.cross_attn_norm = True
i2v_1_3B.eps = 1e-6
