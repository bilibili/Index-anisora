import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


import imageio
import numpy as np
from typing import List
from io import BytesIO
from PIL import Image
import subprocess
from time import sleep
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType, FullStateDictConfig,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict
)

_INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP = None, None
_LOCAL_RANK, _LOCAL_WORLD_SIZE = -1, -1

def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()
    return gif_bytes


def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))


def images_to_mp4_bytes(images: List[Image.Image], duration: float = 1000) -> bytes:
    with BytesIO() as output_buffer:
        with imageio.get_writer(output_buffer, format='mp4', fps=1 / (duration / 1000)) as writer:
            for img in images:
                writer.append_data(np.array(img))
        mp4_bytes = output_buffer.getvalue()
    return mp4_bytes


def save_as_mp4(images: List[Image.Image], file_path: str, duration: float = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_mp4_bytes(images, duration))



def get_local_rank() -> int:
    return _LOCAL_RANK


def get_local_world_size() -> int:
    return _LOCAL_WORLD_SIZE


def _setup_dist_env_from_slurm(args):
    while not os.environ.get("MASTER_ADDR", ""):
        try:
            os.environ["MASTER_ADDR"] = subprocess.check_output(
                "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" %
                os.environ['SLURM_NODELIST'],
                shell=True,
            ).decode().strip()
        except:
            pass
        sleep(1)
    os.environ["MASTER_PORT"] = str(int(args.master_port)+1)
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]

def init_process_groups(args):
    if any([
        x not in os.environ
        for x in ["RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]
    ]):
        _setup_dist_env_from_slurm(args)

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    global _LOCAL_RANK, _LOCAL_WORLD_SIZE
    _LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    _LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    global _INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP
    local_ranks, local_world_sizes = [torch.empty(
        [dist.get_world_size()], dtype=torch.long, device="cuda"
    ) for _ in (0, 1)]
    dist.all_gather_into_tensor(local_ranks, torch.tensor(get_local_rank(), device="cuda"))
    dist.all_gather_into_tensor(local_world_sizes, torch.tensor(get_local_world_size(), device="cuda"))
    local_ranks, local_world_sizes = local_ranks.tolist(), local_world_sizes.tolist()

    node_ranks = [[0]]
    for i in range(1, dist.get_world_size()):
        if len(node_ranks[-1]) == local_world_sizes[i - 1]:
            node_ranks.append([])
        else:
            assert local_world_sizes[i] == local_world_sizes[i - 1]
        node_ranks[-1].append(i)
    for ranks in node_ranks:
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            assert _INTRA_NODE_PROCESS_GROUP is None
            _INTRA_NODE_PROCESS_GROUP = group
    assert _INTRA_NODE_PROCESS_GROUP is not None

    if min(local_world_sizes) == max(local_world_sizes):
        for i in range(get_local_world_size()):
            group = dist.new_group(list(range(i, dist.get_world_size(), get_local_world_size())))
            if i == get_local_rank():
                assert _INTER_NODE_PROCESS_GROUP is None
                _INTER_NODE_PROCESS_GROUP = group
        assert _INTER_NODE_PROCESS_GROUP is not None


def get_intra_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, \
        "Intra-node process group is not initialized."
    return _INTRA_NODE_PROCESS_GROUP


def get_inter_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, \
        "Intra- and inter-node process groups are not initialized."
    return _INTER_NODE_PROCESS_GROUP


def save_model_fsdp_only(rank, model, output_folder, filename):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        consolidated_model_state_dict = model.state_dict()
        if rank == 0:
            torch.save(
                consolidated_model_state_dict,
                os.path.join(output_folder, filename),
            )
        del consolidated_model_state_dict
    dist.barrier()


def save_model(rank, model, output_folder, filename):
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    if rank == 0:
        torch.save(state_dict, os.path.join(output_folder, filename))
    del state_dict
    dist.barrier()


def load_model(rank, model, output_folder, filename, strict=True, logger=None):
    if rank == 0:
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(os.path.join(output_folder, filename), map_location="cpu"),
            strict=strict
        )
        if logger is not None:
            logger.info("Model initialization result:")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpected keys: {unexpected_keys}")
    dist.barrier()


def save_optimizer_fsdp_only(model, optimizer, output_folder, filename):
    with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
    ):
        torch.save(optimizer.state_dict(), os.path.join(output_folder, filename))
    dist.barrier()


def load_optimizer_fsdp_only(optimizer, output_folder, filename):
    optimizer.load_state_dict(
        torch.load(os.path.join(output_folder, filename), map_location="cpu")
    )
    dist.barrier()


def save_optimizer(model, optimizer, output_folder, filename):
    state_dict = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        ),
    )
    torch.save(state_dict, os.path.join(output_folder, filename))
    dist.barrier()


def load_optimizer(model, optimizer, output_folder, filename):
    state_dict = torch.load(os.path.join(output_folder, filename), map_location="cpu")
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=state_dict,
        options=StateDictOptions(
            full_state_dict=False,
            strict=True
        ),
    )
    dist.barrier()