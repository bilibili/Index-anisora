# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .utils import divide, unscaled_init_method
from .utils import VocabUtility


def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False, module=None, name=None, self=None):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight, module=module, name=name)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False,
                                device=weight.device)
    init_method(master_weight, module=module, name=name)
    weight_list = self.partition(full_weight=master_weight)
    rank = get_model_parallel_rank()

    with torch.no_grad():
        weight.copy_(weight_list[rank])
    del weight_list
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """
    def __init__(self, num_embeddings, embedding_dim, params_dtype=torch.float, init_method=unscaled_init_method(0.02), skip_init=False, device=torch.device('cpu')):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition,
                                             self.embedding_dim, dtype=params_dtype,
                                             device=device))
        self.weight.model_parallel = True
        # And initialize.
        if not skip_init:
            _initialize_affine_weight(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method, self=self)

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output
    
    def repartition(self):
        assert self.num_embeddings_per_partition == self.num_embeddings
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index
        self.original_weight = self.weight
        self.weight = torch.nn.Parameter(torch.clone(
            self.weight[self.vocab_start_index:self.vocab_end_index],
            ).detach())
        del self.original_weight

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.num_embeddings_per_partition == self.num_embeddings or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        new_weights = []
        for rank in range(new_model_parallel_size):
            vocab_start_index, vocab_end_index = \
                VocabUtility.vocab_range_from_global_vocab_size(
                    self.num_embeddings, rank,
                    new_model_parallel_size)
            weight = torch.clone(
                full_weight[vocab_start_index:vocab_end_index],
                ).detach()
            new_weights.append(weight)
        if flag == 1:
            return new_weights
        else:
            return new_weights, []
    
    def merge(self, new_weights, new_biases):
        self.weight.data.copy_(torch.cat(new_weights))


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers. only used in initialization.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=unscaled_init_method(0.02), stride=1,
                 keep_master_weight_for_test=False, params_dtype=torch.float, module=None, name=None, skip_init=False, device=torch.device('cpu')):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                             self.input_size, dtype=params_dtype,
                                             device=device))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition,dtype=params_dtype, device=device))
            self.bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        if not skip_init:
            self.master_weight = _initialize_affine_weight(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=self.stride, return_master_weight=keep_master_weight_for_test, module=module, name=name, self=self)

    def forward(self, input_):
        # Set up backprop all-reduce, and don't change the input.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        # input_parallel: [seq_len, input_size]
        # weight: [output_size // mp_size, input_size]
        # bias: [output_size // mp_size]
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output
    
    def repartition(self):
        assert self.output_size_per_partition == self.output_size
        self.output_size_per_partition = divide(self.output_size, get_model_parallel_world_size())
        mp_rank = get_model_parallel_rank()
        mp_size = get_model_parallel_world_size()
        self.original_weight = self.weight
        # weight is arranged as [stride0...stride1...stride2] * [input_size], extract non-contiguous parts
        strides = [1]*self.stride if isinstance(self.stride, int) else self.stride # int means equal number of qkv, or ratios
        assert self.weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        factor = self.weight.shape[0] // sum(strides)
        # decompose weight according to strides
        strided_weights, _acm = [], 0
        for i in range(len(strides)):
            strided_weights.append(self.weight[_acm:_acm+factor*strides[i], :].detach())
            _acm += factor*strides[i]
        new_weight = torch.cat([
            strided_weight[
                (strided_weight.shape[0]//mp_size)*mp_rank:
                (strided_weight.shape[0]//mp_size)*(mp_rank+1)
                ]
            for strided_weight in strided_weights
        ], dim=0).contiguous().view(self.output_size_per_partition, self.input_size)
        self.weight = torch.nn.Parameter(new_weight)
        del self.original_weight
        if self.bias is not None and self.bias.numel() != 0:
            self.original_bias = self.bias
            # decompose bias according to strides
            strided_biases, _acm = [], 0
            for i in range(len(strides)):
                strided_biases.append(self.bias[_acm:_acm+factor*strides[i]].detach())
                _acm += factor*strides[i]
            new_bias = torch.cat([
                strided_bias[
                    (strided_bias.shape[0]//mp_size)*mp_rank:
                    (strided_bias.shape[0]//mp_size)*(mp_rank+1)
                    ]
                for strided_bias in strided_biases
            ], dim=0).contiguous().view(self.output_size_per_partition)
            self.bias = torch.nn.Parameter(new_bias)
            del self.original_bias

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.output_size_per_partition == self.output_size or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        output_size_per_partition = divide(self.output_size, new_model_parallel_size)
        new_weights = []
        new_biases = []

        mp_size = new_model_parallel_size
        # weight is arranged as [stride0...stride1...stride2] * [input_size], extract non-contiguous parts
        strides = [1]*self.stride if isinstance(self.stride, int) else self.stride # int means equal number of qkv, or ratios
        assert full_weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        factor = full_weight.shape[0] // sum(strides)
        # decompose weight according to strides
        strided_weights, _acm = [], 0
        for i in range(len(strides)):
            strided_weights.append(full_weight[_acm:_acm+factor*strides[i], :].detach())
            _acm += factor*strides[i]

        if flag == 2 and self.bias is not None and self.bias.numel() != 0:
            # decompose bias according to strides
            strided_biases, _acm = [], 0
            for i in range(len(strides)):
                strided_biases.append(self.bias[_acm:_acm+factor*strides[i]].detach())
                _acm += factor*strides[i]

        for rank in range(new_model_parallel_size):
            mp_rank = rank
            new_weight = torch.cat([
                strided_weight[
                    (strided_weight.shape[0]//mp_size)*mp_rank:
                    (strided_weight.shape[0]//mp_size)*(mp_rank+1)
                    ]
                for strided_weight in strided_weights
            ], dim=0).contiguous().view(output_size_per_partition, self.input_size)
            new_weights.append(torch.clone(new_weight).detach())
            if flag == 2 and self.bias is not None and self.bias.numel() != 0:
                new_bias = torch.cat([
                    strided_bias[
                        (strided_bias.shape[0]//mp_size)*mp_rank:
                        (strided_bias.shape[0]//mp_size)*(mp_rank+1)
                        ]
                    for strided_bias in strided_biases
                ], dim=0).contiguous().view(output_size_per_partition)
                new_biases.append(torch.clone(new_bias).detach())
        if flag == 1:
            return new_weights
        else:
            return new_weights, new_biases
    
    def merge(self, new_weights, new_biases):
        strides = [1]*self.stride if isinstance(self.stride, int) else self.stride # int means equal number of qkv, or ratios
        assert self.weight.shape[0] % sum(strides) == 0, 'cannot divide weight evenly'
        all_weights = []
        _acm = 0
        for stride in strides:
            for weight in new_weights:
                factor = weight.shape[0] // sum(strides)
                all_weights.append(weight[_acm:_acm+factor*stride])
            _acm += factor*stride
        self.weight.data.copy_(torch.cat(all_weights))
        if self.bias is not None and self.bias.numel() != 0:
            all_biases = []
            _acm = 0
            for stride in strides:
                for bias in new_biases:
                    factor = bias.shape[0] // sum(strides)
                    all_biases.append(bias[_acm:_acm+factor*stride])
                _acm += factor*stride
            self.bias.data.copy_(torch.cat(all_biases))


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=unscaled_init_method(0.02), stride=1,
                 keep_master_weight_for_test=False, params_dtype=torch.float, module=None, name=None, skip_init=False, device=torch.device('cpu'), final_bias=True):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.final_bias = final_bias

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.output_size,
                                             self.input_size_per_partition, dtype=params_dtype, device=device))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype, device=device))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        if not skip_init:
            self.master_weight = _initialize_affine_weight(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test, module=module, name=name, self=self)

    def forward(self, input_):
        # Split the input vector along the last dimension.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        # input_parallel: [seq_len, input_size // mp_size]
        # weight: [output_size, input_size // mp_size]
        if self.final_bias or self.bias is None:
            output_parallel = F.linear(input_parallel, self.weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight, self.bias / get_model_parallel_world_size())
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.final_bias and self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output

    def repartition(self):
        assert self.input_size_per_partition == self.input_size
        self.input_size_per_partition = divide(self.input_size, get_model_parallel_world_size())
        mp_rank = get_model_parallel_rank()
        self.original_weight = self.weight
        self.weight = torch.nn.Parameter(torch.clone(
            self.weight[:, mp_rank*self.input_size_per_partition
                            :(mp_rank+1)*self.input_size_per_partition],
            ).detach())
        del self.original_weight

    def partition(self, new_model_parallel_size=None, full_weight=None):
        assert self.input_size_per_partition == self.input_size or full_weight is not None
        flag = 1
        if full_weight is None:
            full_weight = self.weight
            flag = 2
        if new_model_parallel_size is None:
            new_model_parallel_size = get_model_parallel_world_size()
        input_size_per_partition = divide(self.input_size, new_model_parallel_size)
        new_weights = []
        new_biases = []
        for rank in range(new_model_parallel_size):
            mp_rank = rank
            weight = torch.clone(
                full_weight[:, mp_rank*input_size_per_partition
                                :(mp_rank+1)*input_size_per_partition],
                ).detach()
            new_weights.append(weight)
            if flag == 2 and self.bias is not None and self.bias.numel() != 0:
                new_biases.append(torch.clone(self.bias.data).detach())
        if flag == 1:
            return new_weights
        else:
            return new_weights, new_biases
    
    def merge(self, new_weights, new_biases):
        self.weight.data.copy_(torch.cat(new_weights, 1))
        if self.bias is not None and self.bias.numel() != 0:
            self.bias.data.copy_(new_biases[0])