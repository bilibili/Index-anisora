"""
In this mixin, I use a different implementation than sat/model/finetune/lora.py
I just use a fake linear layer to replace any model with lora mixin.
"""

import torch
import torch.nn as nn
from sat.model.base_model import BaseMixin
import math
from sat.helpers import print_all, print_rank0
from sat.model.transformer import RowParallelLinear, ColumnParallelLinear
from sat.mpu.layers import copy_to_model_parallel_region
from sat import mpu

class HackLinear(nn.Linear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

class HackRowParallelLinear(RowParallelLinear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

class HackColumnParallelLinear(ColumnParallelLinear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

try:
    from bitsandbytes.nn import LinearNF4
    def copy_nested_list(src, dst):
        for i in range(len(dst)):
            if type(dst[i]) is torch.Tensor:
                dst[i].copy_(src[i])
            elif type(dst[i]) is list:
                copy_nested_list(src[i], dst[i])
            else:
                dst[i] = src[i]
    class HackLinearNF4(LinearNF4):
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if prefix + 'weight' in state_dict:
                self.weight.data.copy_(state_dict[prefix+'weight'])
                if self.weight.data.dtype == torch.uint8:
                    copy_nested_list(state_dict[prefix+'quant_state'], self.weight.quant_state)
            if prefix + 'bias' in state_dict:
                self.bias.data.copy_(state_dict[prefix+'bias'])
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            destination[prefix+'quant_state'] = self.weight.quant_state
except Exception as exception:
    print_all("Failed to load bitsandbytes:" + str(exception), level='WARNING')


class HackParameterList(nn.ParameterList):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for i in range(len(self)):
            if prefix + str(i) in state_dict:
                self[i].data.copy_(state_dict[prefix+str(i)])

map_cls = {
    nn.Linear: (HackLinear, {}),
    ColumnParallelLinear: (HackColumnParallelLinear, {'gather_output': False}),
    RowParallelLinear: (HackRowParallelLinear, {'input_is_parallel': True})
}

class LoraLinear(nn.Module):
    def __init__(self, original_cls, partition, in_dim, out_dim, r, lora_alpha=1., lora_dropout=0., qlora=False, original_obj=None):
        super().__init__()
        assert original_obj is not None, "original linear object must be given!"
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        bias = original_obj.bias is not None
        if qlora:
            try:
                self.original = HackLinearNF4(in_dim, out_dim, bias=bias)
            except:
                raise Exception('Build 4bit layer failed. You need to install the latest bitsandbytes. Try `pip install bitsandbytes`. If you still meet error after installation, try running `from bitsandbytes.nn import LinearNF4` with python and fix the error.')
        else:
            base_cls, kwargs = map_cls[original_cls]
            dtype = original_obj.weight.dtype
            if original_cls is ColumnParallelLinear:
                kwargs['stride'] = partition
                kwargs['skip_init'] = True
                kwargs['params_dtype'] = dtype
            elif original_cls is RowParallelLinear:
                kwargs['final_bias'] = original_obj.final_bias
                kwargs['skip_init'] = True
                kwargs['params_dtype'] = dtype
            else:
                kwargs['dtype'] = dtype
            self.original = base_cls(in_dim, out_dim, **kwargs, bias=bias)
        self.original.weight.data.copy_(original_obj.weight.data.detach().clone())
        if bias:
            self.original.bias.data.copy_(original_obj.bias.data.detach().clone())
        if type(partition) is int:
            self.matrix_A = HackParameterList([nn.Parameter(torch.empty((r, original_obj.weight.shape[1]), dtype=dtype)) for _ in range(partition)])
            self.matrix_B = HackParameterList([nn.Parameter(torch.empty((original_obj.weight.shape[0] // partition, r), dtype=dtype)) for _ in range(partition)])
            for i in range(partition):
                nn.init.kaiming_uniform_(self.matrix_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.matrix_B[i])
                self.matrix_B[i].model_parallel = True
        else:
            new_sizes = [original_obj.weight.shape[0] // sum(partition) * i for i in partition]
            self.matrix_A = HackParameterList([nn.Parameter(torch.empty((r, original_obj.weight.shape[1]), dtype=dtype)) for _ in partition])
            self.matrix_B = HackParameterList([nn.Parameter(torch.empty((sz, r), dtype=dtype)) for sz in new_sizes])
            for i in range(len(partition)):
                nn.init.kaiming_uniform_(self.matrix_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.matrix_B[i])
                self.matrix_B[i].model_parallel = True
        self.partition = partition

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # This is not a perfect version, becuase it doesn't handle errors and unexpected keys.
        if prefix + 'weight' in state_dict:
            # load from normal Linear
            self.original._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        else:
            # load from LoraLinear
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            
    def forward(self, x):
        mixed_raw_layer = self.original(x)
        x = self.lora_dropout(x)
        lora_outputs = []
        for mA, mB in zip(self.matrix_A, self.matrix_B):
            lora_outputs.append((copy_to_model_parallel_region(x @ mA.T) @ mB.T) * self.scaling)
        mixed_raw_layer = mixed_raw_layer + torch.cat(lora_outputs, -1)

        return mixed_raw_layer


def replace_linear_with_lora(lin, partition, r, *args, **kw_args):
    if kw_args.get('in_size', None) is not None:
        in_size = kw_args.pop('in_size')
        out_size = kw_args.pop('out_size')
        if out_size is None:
            out_size = in_size * partition
        out_dim, in_dim = out_size , in_size
    else:
        out_dim, in_dim = lin.weight.shape
    original_cls = type(lin)
    new_layer = LoraLinear(original_cls, partition, in_dim, out_dim, r, *args, **kw_args, original_obj=lin)
    device = lin.weight.device
    del lin
    return new_layer.to(device)

def merge_linear_lora(lin):
    if lin.original.weight.data.dtype is not torch.uint8:
        weight = lin.original.weight
        out_dim, in_dim = weight.shape
        new_lin = nn.Linear(in_dim, out_dim, dtype=weight.data.dtype, bias=lin.original.bias is not None)
    else:
        import bitsandbytes.functional as F
        weight = F.dequantize_fp4(lin.original.weight.data, lin.original.weight.quant_state).to(lin.original.bias.data.dtype)
        out_dim, in_dim = weight.shape
        new_lin = HackLinearNF4(in_dim, out_dim, bias=lin.original.bias is not None)
    if lin.original.bias is not None:
        new_lin.bias.data = lin.original.bias.data
    new_qkv = []
    for mA, mB in zip(lin.matrix_A, lin.matrix_B):
        new_qkv.append(mB.data.float() @ mA.data.float() * lin.scaling)
    new_qkv = torch.cat(new_qkv, -2)
    guess_type = lin.original.bias.data.dtype if lin.original.bias is not None else lin.original.weight.data.dtype
    if guess_type is torch.uint8:
        guess_type = torch.float32
    new_lin.weight.data = (weight + new_qkv).to(guess_type)
    return new_lin.cuda() if torch.cuda.is_available() else new_lin

class LoraMixin(BaseMixin):
    def __init__(self, 
                layer_num,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.,
                layer_range = None,
                qlora = False,
                cross_attention = True):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if layer_range is None:
            layer_range = [i for i in range(layer_num)]
        self.layer_range = layer_range

        self.scaling = self.lora_alpha / self.r
        self.qlora = qlora
        self.cross_attention = cross_attention

    def reinit(self, parent_model):
        for i in self.layer_range:
            print_rank0(f'replacing layer {i} attention with lora')
            parent_model.transformer.layers[i].attention.dense = replace_linear_with_lora(parent_model.transformer.layers[i].attention.dense, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora, in_size=parent_model.transformer.hidden_size, out_size=None)
            parent_model.transformer.layers[i].attention.query_key_value = replace_linear_with_lora(parent_model.transformer.layers[i].attention.query_key_value, parent_model.transformer.layers[i].attention.stride, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora, in_size=parent_model.transformer.hidden_size, out_size=None if not parent_model.transformer.num_multi_query_heads else parent_model.transformer.layers[i].attention.inner_hidden_size + parent_model.transformer.layers[i].attention.hidden_size_per_attention_head * parent_model.transformer.layers[i].attention.num_multi_query_heads * 2)
            if self.cross_attention and parent_model.transformer.layers[i].is_decoder:
                print_rank0(f'replacing layer {i} cross attention with lora')
                kv_size = parent_model.transformer.layers[i].cross_attention.inner_hidden_size * 2 if not parent_model.transformer.cross_num_multi_query_heads else parent_model.transformer.layers[i].cross_attention.hidden_size_per_attention_head * parent_model.transformer.layers[i].cross_attention.cross_num_multi_query_heads * 2
                parent_model.transformer.layers[i].cross_attention.dense = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.dense, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora, in_size=parent_model.transformer.layers[i].cross_attention.inner_hidden_size, out_size=parent_model.transformer.hidden_size)
                parent_model.transformer.layers[i].cross_attention.query = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.query, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora, in_size=parent_model.transformer.hidden_size, out_size=parent_model.transformer.layers[i].cross_attention.inner_hidden_size)
                parent_model.transformer.layers[i].cross_attention.key_value = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.key_value, 2, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora, in_size=parent_model.transformer.layers[i].cross_attention.cross_attn_hidden_size, out_size=kv_size)
        if self.qlora:
            print_rank0('replacing chatglm linear layer with 4bit')
            def replace_linear_with_nf4(model, name=None, cache={}):
                if type(model) in (nn.Linear, RowParallelLinear, ColumnParallelLinear):
                    out_dim, in_dim = model.weight.shape
                    bias = model.bias is not None
                    new_linear = HackLinearNF4(in_dim, out_dim, bias=bias)
                    new_linear.weight.data.copy_(model.weight.data.detach().clone())
                    if bias:
                        new_linear.bias.data.copy_(model.bias.data.detach().clone())
                    return new_linear
                names = set()
                for name, child in model.named_children():
                    if name not in names:
                        if child in cache:
                            new_child = cache[child]
                        else:
                            new_child = replace_linear_with_nf4(child, name=name, cache=cache)
                            cache[child] = new_child
                        setattr(model, name, new_child)
                        names.add(name)
                flag = True
                while flag:
                    flag = False
                    for name, child in model.named_children():
                        if name not in names:
                            setattr(model, name, cache[child])
                            names.add(name)
                            flag = True
                return model
            replace_linear_with_nf4(parent_model.transformer, None, {})

    def merge_lora(self):
        for i in self.layer_range:
            print_rank0(f'merge layer {i} lora attention back to linear')
            self.transformer.layers[i].attention.dense = merge_linear_lora(self.transformer.layers[i].attention.dense)
            self.transformer.layers[i].attention.query_key_value = merge_linear_lora(self.transformer.layers[i].attention.query_key_value)
            if self.cross_attention and self.transformer.layers[i].is_decoder:
                print_rank0(f'merge layer {i} lora cross attention back to linear')
                self.transformer.layers[i].cross_attention.dense = merge_linear_lora(self.transformer.layers[i].cross_attention.dense)
                self.transformer.layers[i].cross_attention.query = merge_linear_lora(self.transformer.layers[i].cross_attention.query)
                self.transformer.layers[i].cross_attention.key_value = merge_linear_lora(self.transformer.layers[i].cross_attention.key_value)
