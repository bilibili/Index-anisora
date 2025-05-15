# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import os

# DeepSpeed Team

import torch
import torch.distributed as dist
from fastvideo.utils.parallel_states import nccl_info
from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module


def broadcast(input_: torch.Tensor):
    src = nccl_info.group_id * nccl_info.sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)


def _all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(nccl_info.group, input_, scatter_dim, gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.sp_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.sp_size
        rank = nccl_info.rank_within_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)


def prepare_sequence_parallel_data(
    hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask
):###not use fastvideo default sp data
    return (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    )
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    def prepare(
        hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask
    ):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        attention_mask = all_to_all(attention_mask, scatter_dim=1, gather_dim=0)
        encoder_attention_mask = all_to_all(
            encoder_attention_mask, scatter_dim=1, gather_dim=0
        )
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    sp_size = nccl_info.sp_size
    # frame = hidden_states.shape[2]
    # print(2333333,frame)#13
    # assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
    )

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask


def sp_parallel_dataloader_wrapper(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask = data_item
            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask
            else:
                latents, cond, attn_mask, cond_mask = prepare_sequence_parallel_data(
                    latents, cond, attn_mask, cond_mask
                )
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    attention_mask = attn_mask[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        attention_mask,
                        encoder_attention_mask,
                    )



def _split_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    if pad > 0:
        pad_size = list(input_.shape)
        pad_size[dim] = pad
        input_ = torch.cat([input_, torch.zeros(pad_size, dtype=input_.dtype, device=input_.device)], dim=dim)

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, f"dim_size ({dim_size}) is not divisible by world_size ({world_size})"

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()
    # if output.grad!=None:####must be None...
    #     print(1111111,output.grad)
    return output


def _gather_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type == "cuda"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim)

    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Gather the input sequence.

    Args:
        input_: input matrix.
        process_group: process group.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _gather_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None



class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split sequence.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _split_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)
        return _gather_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None


# def split_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
#     return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, pad)
# def gather_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
#     return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, pad)

# if_print=0
def split_sequence(input_, dim, grad_scale=1.0, pad=0):
    # global if_print
    # if if_print==0:
    #     # print(123232323, int(os.getenv("RANK", "0")), nccl_info.group)
    #     print(123232323, int(os.getenv("RANK", "0")), dist.get_rank(nccl_info.group),dist.get_world_size(nccl_info.group))
    #     if_print=1
    process_group=nccl_info.group
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, pad)
def gather_sequence(input_, dim, grad_scale=1.0, pad=0):
    process_group=nccl_info.group
    # print(process_group)
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, pad)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

def _all_to_all_func(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll1(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)

        return _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll1.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


# def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
#     return _AllToAll1.apply(input_, process_group, scatter_dim, gather_dim)
def all_to_all_comm(input_,scatter_dim=2, gather_dim=1):
    process_group=nccl_info.group
    return _AllToAll1.apply(input_, process_group, scatter_dim, gather_dim)
