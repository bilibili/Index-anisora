import os
import random

import imageio
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
import pdb


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def set_seed(seed, dp_rank=None):
    if seed == -1:
        seed = random.randint(0, 1000000)

    if dp_rank is not None:
        #pdb.set_trace()
        seed = torch.tensor(seed, dtype=torch.int64).cuda()
        if dist.get_world_size() > 1:
            dist.broadcast(seed, 0)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(dp_rank)
        seed = seed + dp_rank

    seed = int(seed)
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


def batch_func(func, *args):
    """
    Apply a function to each element of a batch.
    """
    batch = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.shape[0] == 2:
            batch.append(func(arg))
        else:
            batch.append(arg)

    return batch


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


def save_video(video, output_path, fps):
    """
    Save a video to disk.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, video, fps=fps)
