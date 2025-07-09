# -*- encoding: utf-8 -*-
# @File    :   adapter.py
# @Time    :   2022/6/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
from sat.model.base_model import BaseModel, BaseMixin, non_conflict
import torch.nn as nn
class AdapterMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, adapter_hidden):
        super().__init__()
        self.ff1 = nn.ModuleList([
            nn.Linear(hidden_size, adapter_hidden) for _ in range(num_layers)
        ])
        self.ff2 = nn.ModuleList([
            nn.Linear(adapter_hidden, hidden_size) for _ in range(num_layers)
        ])
        self.ff3 = nn.ModuleList([
            nn.Linear(hidden_size, adapter_hidden) for _ in range(num_layers)
        ])
        self.ff4 = nn.ModuleList([
            nn.Linear(adapter_hidden, hidden_size) for _ in range(num_layers)
        ])

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        attention_output = attention_output + self.ff2[kw_args['layer_id']](nn.functional.gelu(self.ff1[kw_args['layer_id']](attention_output)))

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)
        mlp_output = mlp_output + self.ff4[kw_args['layer_id']](nn.functional.gelu(self.ff3[kw_args['layer_id']](mlp_output)))

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output

    def reinit(self, parent_model=None):
        # refer to https://github.com/google-research/adapter-bert/blob/1a31fc6e92b1b89a6530f48eb0f9e1f04cc4b750/modeling.py#L321
        for ly in self.ff1:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff2:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff3:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff4:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)