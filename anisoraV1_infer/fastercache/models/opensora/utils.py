# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------

import os
from collections.abc import Iterable

import torch
import torch.distributed as dist
from colossalai.checkpoint_io import GeneralCheckpointIO
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torchvision.datasets.utils import download_url

from fastercache.utils.utils import get_logger

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": hf_endpoint + "/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
    "PixArt-Sigma-XL-2-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-256x256.pth",
    "PixArt-Sigma-XL-2-512-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-512-MS.pth",
    "PixArt-Sigma-XL-2-1024-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-1024-MS.pth",
    "PixArt-Sigma-XL-2-2K-MS.pth": hf_endpoint + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-2K-MS.pth",
}


def load_from_sharded_state_dict(model, ckpt_path, model_name="model", strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)


def reparameter(ckpt, name=None, model=None):
    model_name = name
    name = os.path.basename(name)
    if not dist.is_initialized() or dist.get_rank() == 0:
        get_logger().info("loading pretrained model: %s", model_name)
    if name in ["DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"]:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    if name in ["Latte-XL-2-256x256-ucf101.pt"]:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    if name in [
        "PixArt-XL-2-256x256.pth",
        "PixArt-XL-2-SAM-256x256.pth",
        "PixArt-XL-2-512x512.pth",
        "PixArt-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-256x256.pth",
        "PixArt-Sigma-XL-2-512-MS.pth",
        "PixArt-Sigma-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-2K-MS.pth",
    ]:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    if name in [
        "PixArt-1B-2.pth",
    ]:
        ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if ckpt["y_embedder.y_embedding"].shape[0] < model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Extend y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            additional_length = model.y_embedder.y_embedding.shape[0] - ckpt["y_embedder.y_embedding"].shape[0]
            new_y_embedding = torch.zeros(additional_length, model.y_embedder.y_embedding.shape[1])
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat([ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0)
        elif ckpt["y_embedder.y_embedding"].shape[0] > model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Shrink y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            ckpt["y_embedder.y_embedding"] = ckpt["y_embedder.y_embedding"][: model.y_embedder.y_embedding.shape[0]]
    # stdit3 special case
    if type(model).__name__ == "STDiT3" and "PixArt-Sigma" in name:
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            if "blocks." in key:
                ckpt[key.replace("blocks.", "spatial_blocks.")] = ckpt[key]
                del ckpt[key]

    return ckpt


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model_ckpt = download_model(model_name)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def download_model(model_name=None, local_path=None, url=None):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if model_name is not None:
        assert model_name in pretrained_models
        local_path = f"pretrained_models/{model_name}"
        web_path = pretrained_models[model_name]
    else:
        assert local_path is not None
        assert url is not None
        web_path = url
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        dir_name = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        download_url(web_path, dir_name, file_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_checkpoint(model, ckpt_path, save_as_pt=False, model_name="model", strict=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path, model=model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path, model_name, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            get_logger().info("Model checkpoint saved to %s", save_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)
