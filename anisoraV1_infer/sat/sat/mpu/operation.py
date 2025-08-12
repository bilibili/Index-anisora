# -*- encoding: utf-8 -*-
'''
@File    :   operation.py
@Time    :   2023/06/21 17:05:39
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
from sat.mpu import destroy_model_parallel, initialize_model_parallel
from sat.mpu import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

def mp_split_checkpoint(path):
    raise NotImplementedError

def mp_merge_checkpoint(path):
    raise NotImplementedError

def mp_split_model(model, new_model_parallel_size):
    """
    This function makes partitions in-place for a model.
    It takes less memory when world size is small.
    """
    from sat.model.transformer import SelfAttention, CrossAttention

    destroy_model_parallel()
    initialize_model_parallel(new_model_parallel_size)
    def iter_repartition(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, 
                                       SelfAttention, CrossAttention)):
                sub_module.repartition()
            iter_repartition(sub_module)
    iter_repartition(model)

from .initialize import get_node_group, get_node_src_rank, get_node_world_size
from .initialize import get_model_parallel_group, get_model_parallel_src_rank, get_model_parallel_world_size

def mp_split_model_rank0(model, model_full, use_node_group=True):
    """
    This function loads partitions from rank 0.
    It takes less memory when world size is large.
    """
    group = get_node_group() if use_node_group else get_model_parallel_group()
    src = get_node_src_rank() if use_node_group else get_model_parallel_src_rank()
    local_world_size = get_node_world_size() if use_node_group else get_model_parallel_world_size()
    def iter_repartition(new_model, module):
        for (new_name, sub_new_model), (name, sub_module) in zip(new_model.named_children(), module.named_children()):
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
                new_weights, new_biases = sub_module.partition()
                for i in range(local_world_size):
                    if i == 0:
                        sub_new_model.weight.data.copy_(new_weights[src%len(new_weights)])
                    else:
                        torch.distributed.send(new_weights[(src+i)%len(new_weights)].cuda(), src+i)
                if new_biases:
                    for i in range(local_world_size):
                        if i == 0:
                            sub_new_model.bias.data.copy_(new_biases[src%len(new_weights)])
                        else:
                            torch.distributed.send(new_biases[(src+i)%len(new_biases)].cuda(), src+i)
            else:
                for (nn, np), (n, p) in zip(sub_new_model.named_parameters(recurse=False), sub_module.named_parameters(recurse=False)):
                    np.data.copy_(torch.clone(p.data).detach())
                    torch.distributed.broadcast(np.data, src, group=group)
            iter_repartition(sub_new_model, sub_module)
    iter_repartition(model, model_full)

def mp_split_model_receive(model, use_node_group=True):
    group = get_node_group() if use_node_group else get_model_parallel_group()
    src = get_node_src_rank() if use_node_group else get_model_parallel_src_rank()
    def iter_repartition(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, VocabParallelEmbedding):
                torch.distributed.recv(sub_module.weight.data, src)
            elif isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear)):
                torch.distributed.recv(sub_module.weight.data, src)
                if sub_module.bias is not None and sub_module.bias.numel() != 0:
                    torch.distributed.recv(sub_module.bias.data, src)
            else:
                for n, p in sub_module.named_parameters(recurse=False):
                    torch.distributed.broadcast(p.data, src, group=group)
            iter_repartition(sub_module)
    iter_repartition(model)
    
def mp_merge_model_rank0(model, model_full):
    assert get_model_parallel_world_size() == torch.distributed.get_world_size(), "Merging model is only supported for model_parallel_size == world_size!"
    def iter_merge(new_model, module):
        for (new_name, sub_new_model), (name, sub_module) in zip(new_model.named_children(), module.named_children()):
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
                new_weights, new_biases = sub_module.partition()
                new_weights = [x.to(sub_new_model.weight.device) for x in new_weights]
                torch.distributed.gather(sub_new_model.weight.data, gather_list=new_weights, dst=0)
                if new_biases:
                    new_biases = [x.to(sub_new_model.weight.device) for x in new_biases]
                    torch.distributed.gather(sub_new_model.bias.data, gather_list=new_biases, dst=0)
                sub_module.merge([torch.clone(x.cpu()).detach() for x in new_weights], [torch.clone(x.cpu()).detach() for x in new_biases])
                del new_weights
                if new_biases:
                    del new_biases
            else:
                for (nn, np), (n, p) in zip(sub_new_model.named_parameters(recurse=False), sub_module.named_parameters(recurse=False)):
                    p.data.copy_(torch.clone(np.data.cpu()).detach())
            iter_merge(sub_new_model, sub_module)
    iter_merge(model, model_full)

def mp_merge_model_send(model):
    assert get_model_parallel_world_size() == torch.distributed.get_world_size(), "Merging model is only supported for model_parallel_size == world_size!"
    def iter_merge(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, VocabParallelEmbedding):
                torch.distributed.gather(sub_module.weight.data, dst=0)
            elif isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear)):
                torch.distributed.gather(sub_module.weight.data, dst=0)
                if sub_module.bias is not None and sub_module.bias.numel() != 0:
                    torch.distributed.gather(sub_module.bias.data, dst=0)
            iter_merge(sub_module)
    iter_merge(model)