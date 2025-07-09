# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2022/06/03 23:01:46
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

# dynamic import according to name with importlib
from importlib import import_module

avaliable_ops = {
    'LayerNorm': 'sat.ops.layernorm',
    'f_similar': 'sat.ops.local_attention_function',
    'f_weighting': 'sat.ops.local_attention_function',
    'FusedScaleMaskSoftmax': 'sat.ops.scaled_mask_softmax',
    'FusedEmaAdam': 'sat.ops.fused_ema_adam',
    'memory_efficient_attention': 'sat.ops.memory_efficient_attention',
    'AdamW': 'sat.ops.npu_adamw',
}

for name, path in avaliable_ops.items():
    # define some objects with the same name as the ops
    # so that we can use them as a placeholder
    # when __call__() is called, the real ops will be imported and called
    locals()[name] = type(name + 'Shell', (object,), {
        '__init__': lambda self, *args, **kwargs: None,
        '__call__': lambda self, *args, **kwargs: getattr(import_module(self.path), self.name)(*args, **kwargs),
        'name': name,
        'path': path,
    })()

