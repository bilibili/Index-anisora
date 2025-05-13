# -*- encoding: utf-8 -*-
# @File    :   lora.py
# @Time    :   2022/6/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
# -*- encoding: utf-8 -*-
'''
@File    :   prompt_tuning.py
@Time    :   2021/12/12 20:45:18
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn as nn
from sat.model.transformer import standard_attention
from sat.model.base_model import BaseModel, BaseMixin, non_conflict
from sat.mpu.utils import split_tensor_along_last_dim
import torch.nn.functional as F

class LoRAMixin(BaseMixin):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int = 24,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            layer_range = None,
    ):
        super().__init__()
        # Actual trainable parameters
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        if layer_range is None:
            layer_range = [i for i in range(layer_num)]
        self.layer_range = layer_range
        self.lora_linear = nn.ModuleList([
            nn.ParameterDict()
            for layer_id in range(layer_num)
        ])
        matrices = ["Q", "K", "V", "O"]

        for i in layer_range:
            for matrix in matrices:
                self.lora_linear[i][matrix+"_A"] = nn.Parameter(torch.zeros((r, hidden_size)))
                self.lora_linear[i][matrix+"_B"] = nn.Parameter(torch.zeros((hidden_size, r)))
                nn.init.kaiming_uniform_(self.lora_linear[i][matrix+"_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i][matrix+"_B"])


        self.scaling = self.lora_alpha / self.r


    def attention_forward(self, hidden_states, mask, layer_id, **kw_args):
        attention_fn = standard_attention
        if 'attention_fn' in self.transformer.hooks:
            attention_fn = self.transformer.hooks['attention_fn']
        layer = self.transformer.layers[layer_id].attention
        lora_layer = self.lora_linear[layer_id]

        mixed_raw_layer = layer.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        if layer_id in self.layer_range:
            mixed_query_layer = mixed_query_layer + (self.lora_dropout(hidden_states) @ lora_layer["Q_A"].T @ lora_layer["Q_B"].T) * self.scaling
            mixed_key_layer = mixed_key_layer + (self.lora_dropout(hidden_states) @ lora_layer["K_A"].T @ lora_layer["K_B"].T) * self.scaling
            mixed_value_layer = mixed_value_layer + (self.lora_dropout(hidden_states) @ lora_layer["V_A"].T @ lora_layer["V_B"].T) * self.scaling


        dropout_fn = layer.attention_dropout if self.training else None

        query_layer = layer._transpose_for_scores(mixed_query_layer)
        key_layer = layer._transpose_for_scores(mixed_key_layer)
        value_layer = layer._transpose_for_scores(mixed_value_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (layer.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = layer.dense(context_layer)

        if layer_id in self.layer_range:
            output = output + (self.lora_dropout(context_layer) @ lora_layer["O_A"].T @ lora_layer["O_B"].T ) * self.scaling

        if self.training:
            output = layer.output_dropout(output)
        return output