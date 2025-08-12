# -*- encoding: utf-8 -*-
# @File    :   ffadd.py
# @Time    :   2022/6/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
from sat.model.base_model import BaseMixin, non_conflict
import torch
class FFADDMixin(BaseMixin):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int = 24,
            r: int = 0,
            layer_range = None,
    ):
        super().__init__()
        # Actual trainable parameters
        self.r = r

        self.ffadd_linear = nn.ModuleList([
            nn.ModuleList()
            for layer_id in range(layer_num)
        ])

        if layer_range is None:
            layer_range = [i for i in range(layer_num)]
        self.layer_range = layer_range
        for i in layer_range:
            self.ffadd_linear[i].append(torch.nn.Linear(hidden_size, r, bias=True))
            self.ffadd_linear[i].append(torch.nn.Linear(r, hidden_size, bias=True))
            nn.init.zeros_(self.ffadd_linear[i][1].weight)
            nn.init.zeros_(self.ffadd_linear[i][1].bias)


    def mlp_forward(self, hidden_states, layer_id,  attention_output = None, **kw_args):
        layer = self.transformer.layers[layer_id].mlp
        intermediate_parallel = layer.dense_h_to_4h(hidden_states)
        intermediate_parallel = layer.activation_func(intermediate_parallel)
        output = layer.dense_4h_to_h(intermediate_parallel)

        if layer_id in self.layer_range:
            ffadd_layer = self.ffadd_linear[layer_id]
            layer = self.transformer.layers[layer_id].mlp
            intermediate_add = ffadd_layer[0](hidden_states)
            intermediate_add = layer.activation_func(intermediate_add)
            if attention_output is not None:
                kw_args["output_this_layer"]["0"] = intermediate_add.data.cpu().numpy()
            output2 = ffadd_layer[1](intermediate_add)
            output = output + output2

        if self.training:
            output = layer.dropout(output)

        return output