import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def str_to_dtype(x: str):
    if x == "fp32":
        return torch.float32
    elif x == "fp16":
        return torch.float16
    elif x == "bf16":
        return torch.bfloat16
    else:
        raise RuntimeError(f"Only fp32, fp16 and bf16 are supported, but got {x}")


def merge_args(args1, args2):
    """
    Merge two argparse Namespace objects.
    """
    if args2 is None:
        return args1

    for k in args2._content.keys():
        if k in args1.__dict__:
            v = getattr(args2, k)
            if isinstance(v, ListConfig) or isinstance(v, DictConfig):
                v = OmegaConf.to_object(v)
            setattr(args1, k, v)
        else:
            raise RuntimeError(f"Unknown argument {k}")

    return args1


def all_exists(paths):
    return all(os.path.exists(path) for path in paths)


def get_logger():
    return logging.getLogger(__name__)


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def all_to_all(input_: torch.Tensor, gather_dim: int, scatter_dim: int) -> torch.Tensor:
    assert gather_dim != scatter_dim
    assert 0 <= gather_dim < input_.ndim
    assert 0 <= scatter_dim < input_.ndim
    world_size = dist.get_world_size()
    assert input_.size(scatter_dim) % world_size == 0

    if world_size == 1:
        return input_

    inputs = [x.contiguous() for x in input_.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(x) for x in inputs]
    dist.all_to_all(outputs, inputs)

    return torch.cat(outputs, dim=gather_dim)


def sp_split(input_: torch.Tensor) -> torch.Tensor:
    size = dist.get_world_size()
    rank = dist.get_rank()
    # print(111111111*3,size,rank,input_.shape)#888888 4 1 torch.Size([2, 40006, 3072])
    if size == 1:
        return input_
    assert input_.size(1) % size == 0
    return input_.chunk(size, dim=1)[rank].contiguous()


def sp_gather(input_: torch.Tensor) -> torch.Tensor:
    size = dist.get_world_size()
    rank = dist.get_rank()
    if size == 1:
        return input_
    output = [torch.empty_like(input_) for _ in range(size)]
    dist.all_gather(output, input_)
    return torch.cat(output, dim=1)

def _setup_dist_env_from_slurm():
    import subprocess
    from time import sleep
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
    os.environ["MASTER_PORT"] = str(18183)
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]

def init_process_groups():
    if any([
        x not in os.environ
        for x in ["RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]
    ]):
        _setup_dist_env_from_slurm()

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
