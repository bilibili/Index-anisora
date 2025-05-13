
import torch
import logging
import requests
import json
from typing import List
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def load_image(image_file, max_image_size=None):
    post_fixs = [".jpg", ".png", ".jpeg", ".gif"]
    if image_file is None:
        return None
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image_file = Path(image_file)
        if not image_file.exists() and not image_file.is_file():
            if all([not image_file.with_suffix(post_fix).exists() for post_fix in post_fixs]):
                raise FileNotFoundError(f"Cannot find image file {image_file}")
            else:
                for post_fix in post_fixs:
                    if image_file.with_suffix(post_fix).exists():
                        image_file = image_file.with_suffix(post_fix)
                        break
                    
        if not isinstance(image_file, str):
            image_file = str(image_file)
        if image_file.startswith("http"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            import os
            image = Image.open(image_file).convert("RGB")
    
    # control the longest side of the image
    if isinstance(max_image_size, int) and max_image_size > 0 and max(image.size) > max_image_size:
        scale = max_image_size / max(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size)
    return image


def load_images(image_files, max_image_size=None):
    if not isinstance(image_files, list):
        return load_image(image_files, max_image_size)
    out = []
    for image_file in tqdm(image_files, desc="Loading images", disable=len(image_files) < 1000):
        if isinstance(image_file, Image.Image):
            image = image_file
        else:
            image = load_image(image_file, max_image_size)
        out.append(image)
    return out

def load_json_data(data_file):
    """
    Read
    """
    if data_file.endswith(".json"):
        with open(data_file, "r") as f:
            data = json.load(f)
    elif data_file.endswith(".jsonl"):
        with open(data_file, "r") as f:
            lines = f.readlines()
        data = [json.loads(line.strip()) for line in lines]
    else:
        raise ValueError("Unknown json file type")
    return data


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

## from otter train.utils
import os
import random
import subprocess
import sys
from contextlib import suppress

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

try:
    from transformers.models.idefics.processing_idefics import image_attention_mask_for_packed_input_ids, incremental_to_binary_attention_mask
except ImportError:
    print("Failed to import Idefics processing module.")


def truncate_text(path, keep_start=10, keep_end=10, truncate_to="..."):
    if len(path) <= (keep_start + keep_end + len(truncate_to)):
        return path
    return path[:keep_start] + truncate_to + path[-keep_end:]


def master_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == "fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return suppress


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


def get_checkpoint_deepspeed_zero3(args, model):
    state_dict = {}

    for name, p in model.named_parameters():
        if p.requires_grad:
            state_dict[name] = p.data
    return state_dict

    # if torch.distributed.get_rank() == 0:
    #     # 有参数
    #     print(device_id, f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


# supporting idefics processing
def get_image_attention_mask(output_input_ids, max_num_images, tokenizer, include_image=True):
    # image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
    # image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    if include_image:
        image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
        image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    else:
        # in full language mode we set the image mask to all-0s
        image_attention_mask = torch.zeros(output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool)
    return image_attention_mask


def verify_yaml(args):
    if args.rank != 0:
        return

    # Run pytest with the necessary arguments.
    result = subprocess.run(["pytest", "-m", "prerun", f"--yaml-path={args.training_data_yaml}"])

    if result.returncode != 0:
        print("YAML verification failed!")
        sys.exit(1)


def get_grouped_params(model, wd):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

    for n, p in model.named_parameters():
        # if p.requires_grad:
        if apply_decay(n):
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": wd},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def save_checkpoint(epoch, model, args, accelerator, unwrapped_model=None, global_step=None):
    """Save a checkpoint for the model."""
    # Ensure the directory exists
    if not os.path.exists(args.external_save_dir):
        os.makedirs(args.external_save_dir)

    if unwrapped_model is None:
        unwrapped_model = accelerator.unwrap_model(model)

    # Formulate the checkpoint filename based on whether it's an epoch or global_step checkpoint
    if global_step:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_steps_{global_step}.pt"
        checkpoint_dict = {
            "steps": global_step,
            "model_state_dict": get_checkpoint(unwrapped_model),
        }
    else:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_{epoch}.pt"
        checkpoint_dict = {"model_state_dict": get_checkpoint(unwrapped_model)}

    # Save the checkpoint if rank is 0
    if args.rank == 0:
        print(f"Saving checkpoint to {checkpoint_path}")
        accelerator.save(checkpoint_dict, checkpoint_path)

        # Save the model's configuration
        unwrapped_model.config.save_pretrained(args.external_save_dir)

        # Remove the previous checkpoint if required
        if args.delete_previous_checkpoint:
            if global_step:
                prev_checkpoint_path = f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"
                if os.path.exists(prev_checkpoint_path):
                    os.remove(prev_checkpoint_path)
            elif epoch > 0:
                os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")


def save_checkpoint(checkpoint_dict, save_path, is_main_process, save_function):
    """Helper function to save the checkpoint."""
    save_function(checkpoint_dict, f"{save_path}/final_weights.pt", is_main_process=is_main_process)


def save_pretrained(component, save_path, is_main_process, save_function):
    """Helper function to save pretrained components."""
    component.save_pretrained(save_path, is_main_process=is_main_process, save_function=save_function, safe_serialization=False)


def save_final_weights(model, args, accelerator, processor=None, tokenizer=None):
    """Save final weights of the model."""
    unwrapped_model = accelerator.unwrap_model(model)
    is_main_process = accelerator.is_main_process
    save_path = args.external_save_dir
    model_name = args.model_name.lower()

    unwrapped_model.config.save_pretrained(save_path)

    if args.save_hf_model:
        save_pretrained(unwrapped_model, save_path, is_main_process, accelerator.save)

        if "idefics" in model_name or "fuyu" in model_name:
            save_pretrained(processor, save_path, is_main_process, accelerator.save)

        if "llama2" in model_name:
            save_pretrained(tokenizer, save_path, is_main_process, accelerator.save)
    else:
        # Save based on the distributed type
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            checkpoint_dict = accelerator.get_state_dict(model)
        else:
            checkpoint_dict = get_checkpoint(model=unwrapped_model)

        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in trainable_params_name}

        save_checkpoint(checkpoint_dict, save_path, is_main_process, accelerator.save)


def get_weights_for_dataloaders(dataloaders):
    total_samples = sum(len(dataloader.dataset) for dataloader in dataloaders)
    weights = [len(dataloader.dataset) / total_samples for dataloader in dataloaders]
    return weights


def get_next_dataloader(dataloader_iterators, weights):
    chosen_dataloader_index = np.random.choice(len(dataloader_iterators), p=weights)
    return dataloader_iterators[chosen_dataloader_index]


def find_and_remove_tokens(input_tensor, labels_tensor, attention_mask_tensor, token_id, tokenizer):
    batch_size, seq_len = input_tensor.size()

    # Create lists to store the new tensors
    new_input_list = []
    new_labels_list = []
    new_attention_mask_list = []

    # Loop over each sequence in the batch
    for i in range(batch_size):
        single_input = input_tensor[i, :]
        single_label = labels_tensor[i, :]
        single_attention_mask = attention_mask_tensor[i, :]

        # Remove the token_id
        new_single_input = torch.masked_select(single_input, single_input != token_id)
        new_single_label = torch.masked_select(single_label, single_input != token_id)
        new_single_attention_mask = torch.masked_select(single_attention_mask, single_input != token_id)

        # Append the new sequence to the list
        new_input_list.append(new_single_input)
        new_labels_list.append(new_single_label)
        new_attention_mask_list.append(new_single_attention_mask)

    # Pad sequences within each batch to match the longest sequence
    new_input = torch.nn.utils.rnn.pad_sequence(new_input_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels_list, batch_first=True, padding_value=-100)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0)

    return new_input, new_labels, new_attention_mask


def delete_tensors_from_dict(d):
    """Recursively delete tensors from a nested dictionary."""
    keys_to_delete = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            keys_to_delete.append(k)
        elif isinstance(v, list):
            new_list = [item for item in v if not isinstance(item, torch.Tensor)]
            d[k] = new_list
        elif isinstance(v, dict):
            delete_tensors_from_dict(v)
    for key in keys_to_delete:
        del d[key]



import os
import torch


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device
