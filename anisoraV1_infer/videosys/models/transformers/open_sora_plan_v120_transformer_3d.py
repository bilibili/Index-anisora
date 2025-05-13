# Adapted from Open-Sora-Plan

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# --------------------------------------------------------

import collections
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import PixArtAlphaTextProjection, SinusoidalPositionalEmbedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormSingle, AdaLayerNormZero
from diffusers.utils import deprecate, is_torch_version
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from videosys.core.comm import all_to_all_comm, gather_sequence, split_sequence
from videosys.core.pab_mgr import enable_pab, if_broadcast_cross, if_broadcast_spatial
from videosys.core.parallel_mgr import ParallelManager
from videosys.core.pipeline import VideoSysPipelineOutput

torch_npu = None
npu_config = None
set_run_dtype = None


class PositionGetter3D(object):
    """return positions of patches"""

    def __init__(
        self,
    ):
        self.cache_positions = {}

    def __call__(self, b, t, h, w, device):
        if not (b, t, h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]

        return pos


class RoPE3D(torch.nn.Module):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        # for (batch_size x ntokens x nheads x dim)
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)
    grid_t = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / (grid_size[2] / base_size[2]) / interpolation_scale[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    # import ipdb;ipdb.set_trace()
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # import ipdb;ipdb.set_trace()
    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid return: pos_embed: [grid_size, embed_dim] or
    [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        # assert num_frames == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
        )
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = rearrange(latent, "b c t h w -> (b t) c h w")
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                raise NotImplementedError
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        # print('video_latent is None, image_latent is None', video_latent is None, image_latent is None)
        return video_latent, image_latent


class OverlapPatchEmbed3D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        # assert patch_size_t == 1 and patch_size == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size_t, patch_size, patch_size),
            stride=(patch_size_t, patch_size, patch_size),
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
        )
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        # latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)

        if self.flatten:
            # latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
            latent = rearrange(latent, "b c t h w -> (b t) (h w) c ")
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                # import ipdb;ipdb.set_trace()
                # raise NotImplementedError
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        return video_latent, image_latent


class OverlapPatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
    ):
        super().__init__()
        assert patch_size_t == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t
        )
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = rearrange(latent, "b c t h w -> (b t) c h w")
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                # import ipdb;ipdb.set_trace()
                # raise NotImplementedError
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        return video_latent, image_latent


class Attention(Attention_):
    def __init__(self, downsampler, attention_mode, use_rope, interpolation_scale_thw, **kwags):
        processor = AttnProcessor2_0(
            attention_mode=attention_mode, use_rope=use_rope, interpolation_scale_thw=interpolation_scale_thw
        )
        super().__init__(processor=processor, **kwags)
        self.downsampler = None
        if downsampler:  # downsampler  k155_s122
            downsampler_ker_size = list(re.search(r"k(\d{2,3})", downsampler).group(1))  # 122
            down_factor = list(re.search(r"s(\d{2,3})", downsampler).group(1))
            downsampler_ker_size = [int(i) for i in downsampler_ker_size]
            downsampler_padding = [(i - 1) // 2 for i in downsampler_ker_size]
            down_factor = [int(i) for i in down_factor]

            if len(downsampler_ker_size) == 2:
                self.downsampler = DownSampler2d(
                    kwags["query_dim"],
                    kwags["query_dim"],
                    kernel_size=downsampler_ker_size,
                    stride=1,
                    padding=downsampler_padding,
                    groups=kwags["query_dim"],
                    down_factor=down_factor,
                    down_shortcut=True,
                )
            elif len(downsampler_ker_size) == 3:
                self.downsampler = DownSampler3d(
                    kwags["query_dim"],
                    kwags["query_dim"],
                    kernel_size=downsampler_ker_size,
                    stride=1,
                    padding=downsampler_padding,
                    groups=kwags["query_dim"],
                    down_factor=down_factor,
                    down_shortcut=True,
                )

        # parallel
        self.parallel_manager: ParallelManager = None

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask


class DownSampler3d(nn.Module):
    def __init__(self, *args, **kwargs):
        """Required kwargs: down_factor, downsampler"""
        super().__init__()
        self.down_factor = kwargs.pop("down_factor")
        self.down_shortcut = kwargs.pop("down_shortcut")
        self.layer = nn.Conv3d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        x.shape[0]
        x = rearrange(x, "b (t h w) d -> b d t h w", t=t, h=h, w=w)
        if npu_config is None:
            x = self.layer(x) + (x if self.down_shortcut else 0)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.layer, x, x_dtype) + (x if self.down_shortcut else 0)

        self.t = t // self.down_factor[0]
        self.h = h // self.down_factor[1]
        self.w = w // self.down_factor[2]
        x = rearrange(
            x,
            "b d (t dt) (h dh) (w dw) -> (b dt dh dw) (t h w) d",
            t=t // self.down_factor[0],
            h=h // self.down_factor[1],
            w=w // self.down_factor[2],
            dt=self.down_factor[0],
            dh=self.down_factor[1],
            dw=self.down_factor[2],
        )

        attention_mask = rearrange(attention_mask, "b 1 (t h w) -> b 1 t h w", t=t, h=h, w=w)
        attention_mask = rearrange(
            attention_mask,
            "b 1 (t dt) (h dh) (w dw) -> (b dt dh dw) 1 (t h w)",
            t=t // self.down_factor[0],
            h=h // self.down_factor[1],
            w=w // self.down_factor[2],
            dt=self.down_factor[0],
            dh=self.down_factor[1],
            dw=self.down_factor[2],
        )
        return x, attention_mask

    def reverse(self, x, t, h, w):
        x = rearrange(
            x,
            "(b dt dh dw) (t h w) d -> b (t dt h dh w dw) d",
            t=t,
            h=h,
            w=w,
            dt=self.down_factor[0],
            dh=self.down_factor[1],
            dw=self.down_factor[2],
        )
        return x


class DownSampler2d(nn.Module):
    def __init__(self, *args, **kwargs):
        """Required kwargs: down_factor, downsampler"""
        super().__init__()
        self.down_factor = kwargs.pop("down_factor")
        self.down_shortcut = kwargs.pop("down_shortcut")
        self.layer = nn.Conv2d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        x.shape[0]
        x = rearrange(x, "b (t h w) d -> (b t) d h w", t=t, h=h, w=w)
        x = self.layer(x) + (x if self.down_shortcut else 0)

        self.t = 1
        self.h = h // self.down_factor[0]
        self.w = w // self.down_factor[1]

        x = rearrange(
            x,
            "b d (h dh) (w dw) -> (b dh dw) (h w) d",
            h=h // self.down_factor[0],
            w=w // self.down_factor[1],
            dh=self.down_factor[0],
            dw=self.down_factor[1],
        )

        attention_mask = rearrange(attention_mask, "b 1 (t h w) -> (b t) 1 h w", h=h, w=w)
        attention_mask = rearrange(
            attention_mask,
            "b 1 (h dh) (w dw) -> (b dh dw) 1 (h w)",
            h=h // self.down_factor[0],
            w=w // self.down_factor[1],
            dh=self.down_factor[0],
            dw=self.down_factor[1],
        )
        return x, attention_mask

    def reverse(self, x, t, h, w):
        x = rearrange(
            x, "(b t dh dw) (h w) d -> b (t h dh w dw) d", t=t, h=h, w=w, dh=self.down_factor[0], dw=self.down_factor[1]
        )
        return x


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_mode="xformers", use_rope=False, interpolation_scale_thw=(1, 1, 1)):
        self.use_rope = use_rope
        self.interpolation_scale_thw = interpolation_scale_thw
        if self.use_rope:
            self._init_rope(interpolation_scale_thw)
        self.attention_mode = attention_mode
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8,
        height: int = 16,
        width: int = 16,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if attn.downsampler is not None:
            hidden_states, attention_mask = attn.downsampler(hidden_states, attention_mask, t=frame, h=height, w=width)
            frame, height, width = attn.downsampler.t, attn.downsampler.h, attn.downsampler.w

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.parallel_manager.sp_size > 1 and query.shape[2] == key.shape[2]:
            func = lambda x: all_to_all_comm(
                x, process_group=attn.parallel_manager.sp_group, scatter_dim=1, gather_dim=2
            )
            query, key, value = map(func, [query, key, value])

        if self.use_rope:
            # require the shape of (batch_size x nheads x ntokens x dim)
            pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)

        # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
        # 0, 0 ->(bool) False, False ->(any) False ->(not) True
        if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
            attention_mask = None
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if attn.parallel_manager.sp_size > 1 and query.shape[2] == key.shape[2]:
            hidden_states = all_to_all_comm(
                hidden_states, process_group=attn.parallel_manager.sp_group, scatter_dim=2, gather_dim=1
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if attn.downsampler is not None:
            hidden_states = attn.downsampler.reverse(hidden_states, t=frame, h=height, w=width)
        return hidden_states


class FeedForward_Conv3d(nn.Module):
    def __init__(self, downsampler, dim, hidden_features, bias=True):
        super(FeedForward_Conv3d, self).__init__()

        self.bias = bias

        self.project_in = nn.Linear(dim, hidden_features, bias=bias)

        self.dwconv = nn.ModuleList(
            [
                nn.Conv3d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(5, 5, 5),
                    stride=1,
                    padding=(2, 2, 2),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
                nn.Conv3d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
                nn.Conv3d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(1, 1, 1),
                    stride=1,
                    padding=(0, 0, 0),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
            ]
        )

        self.project_out = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x, t, h, w):
        # import ipdb;ipdb.set_trace()
        if npu_config is None:
            x = self.project_in(x)
            x = rearrange(x, "b (t h w) d -> b d t h w", t=t, h=h, w=w)
            x = F.gelu(x)
            out = x
            for module in self.dwconv:
                out = out + module(x)
            out = rearrange(out, "b d t h w -> b (t h w) d", t=t, h=h, w=w)
            x = self.project_out(out)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.project_in, x, npu_config.replaced_type)
            x = rearrange(x, "b (t h w) d -> b d t h w", t=t, h=h, w=w)
            x = F.gelu(x)
            out = x
            for module in self.dwconv:
                out = out + npu_config.run_conv3d(module, x, npu_config.replaced_type)
            out = rearrange(out, "b d t h w -> b (t h w) d", t=t, h=h, w=w)
            x = npu_config.run_conv3d(self.project_out, out, x_dtype)
        return x


class FeedForward_Conv2d(nn.Module):
    def __init__(self, downsampler, dim, hidden_features, bias=True):
        super(FeedForward_Conv2d, self).__init__()

        self.bias = bias

        self.project_in = nn.Linear(dim, hidden_features, bias=bias)

        self.dwconv = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(5, 5),
                    stride=1,
                    padding=(2, 2),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
                nn.Conv2d(
                    hidden_features,
                    hidden_features,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=(0, 0),
                    dilation=1,
                    groups=hidden_features,
                    bias=bias,
                ),
            ]
        )

        self.project_out = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x, t, h, w):
        # import ipdb;ipdb.set_trace()
        x = self.project_in(x)
        x = rearrange(x, "b (t h w) d -> (b t) d h w", t=t, h=h, w=w)
        x = F.gelu(x)
        out = x
        for module in self.dwconv:
            out = out + module(x)
        out = rearrange(out, "(b t) d h w -> b (t h w) d", t=t, h=h, w=w)
        x = self.project_out(out)
        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        attention_mode: str = "xformers",
        downsampler: str = None,
        use_rope: bool = False,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1),
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.downsampler = downsampler

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            attention_mode=attention_mode,
            downsampler=downsampler,
            use_rope=use_rope,
            interpolation_scale_thw=interpolation_scale_thw,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                attention_mode=attention_mode,
                downsampler=False,
                use_rope=False,
                interpolation_scale_thw=interpolation_scale_thw,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        if downsampler:
            downsampler_ker_size = list(re.search(r"k(\d{2,3})", downsampler).group(1))  # 122
            # if len(downsampler_ker_size) == 3:
            #     self.ff = FeedForward_Conv3d(
            #         downsampler,
            #         dim,
            #         2 * dim,
            #         bias=ff_bias,
            #     )
            # elif len(downsampler_ker_size) == 2:
            self.ff = FeedForward_Conv2d(
                downsampler,
                dim,
                2 * dim,
                bias=ff_bias,
            )
        else:
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # pab
        self.spatial_last = None
        self.spatial_count = 0
        self.cross_last = None
        self.cross_count = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        frame: int = None,
        height: int = None,
        width: int = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        org_timestep: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # import ipdb;ipdb.set_trace()
        if self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
        else:
            raise ValueError("Incorrect norm used")

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        broadcast, self.spatial_count = if_broadcast_spatial(int(org_timestep[0]), self.spatial_count)
        if broadcast:
            attn_output = self.spatial_last
        else:
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                frame=frame,
                height=height,
                width=width,
                **cross_attention_kwargs,
            )

            if enable_pab():
                self.spatial_last = attn_output

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            broadcast, self.cross_count = if_broadcast_cross(int(org_timestep[0]), self.cross_count)
            if broadcast:
                attn_output = self.cross_last

            else:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )

                if enable_pab():
                    self.cross_last = attn_output
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # if self._chunk_size is not None:
        #     # "feed_forward_chunk_size" can be used to save memory
        #     ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        # else:

        if self.downsampler:
            ff_output = self.ff(norm_hidden_states, t=frame, h=height, w=width)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class OpenSoraT2V(ModelMixin, ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale_h: float = None,
        interpolation_scale_w: float = None,
        interpolation_scale_t: float = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = "xformers",
        downsampler: str = None,
        use_rope: bool = False,
        use_stable_fp32: bool = False,
    ):
        super().__init__()

        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        # Set some common variables used across the board.
        self.use_rope = use_rope
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale_t = interpolation_scale_t
        self.interpolation_scale_h = interpolation_scale_h
        self.interpolation_scale_w = interpolation_scale_w
        self.downsampler = downsampler
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.config.hidden_size = self.inner_dim
        use_additional_conditions = False
        # if use_additional_conditions is None:
        # if norm_type == "ada_norm_single" and sample_size == 128:
        #     use_additional_conditions = True
        # else:
        # use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        assert in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Initialize the right blocks.
        # Initialize the output blocks and other projection blocks when necessary.
        self._init_patched_inputs(norm_type=norm_type)

        # parallel
        self.parallel_manager: ParallelManager = None

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size_t is not None, "OpenSoraT2V over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraT2V over patched input must provide sample_size"
        # assert not (self.config.sample_size_t == 1 and self.config.patch_size_t == 2), "Image do not need patchfy in t-dim"

        self.num_frames = self.config.sample_size_t
        self.config.sample_size = to_2tuple(self.config.sample_size)
        self.height = self.config.sample_size[0]
        self.width = self.config.sample_size[1]
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        interpolation_scale_t = (
            ((self.config.sample_size_t - 1) // 16 + 1)
            if self.config.sample_size_t % 2 == 1
            else self.config.sample_size_t / 16
        )
        interpolation_scale_t = (
            self.config.interpolation_scale_t
            if self.config.interpolation_scale_t is not None
            else interpolation_scale_t
        )
        interpolation_scale = (
            self.config.interpolation_scale_h
            if self.config.interpolation_scale_h is not None
            else self.config.sample_size[0] / 30,
            self.config.interpolation_scale_w
            if self.config.interpolation_scale_w is not None
            else self.config.sample_size[1] / 40,
        )
        if self.config.downsampler is not None and len(self.config.downsampler) == 9:
            self.pos_embed = OverlapPatchEmbed3D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )
        elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
            self.pos_embed = OverlapPatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )

        else:
            self.pos_embed = PatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )
        interpolation_scale_thw = (interpolation_scale_t, *interpolation_scale)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    attention_mode=self.config.attention_mode,
                    downsampler=self.config.downsampler,
                    use_rope=self.config.use_rope,
                    interpolation_scale_thw=interpolation_scale_thw,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.config.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                self.inner_dim,
                self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels,
            )
        elif self.config.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
            self.proj_out = nn.Linear(
                self.inner_dim,
                self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels,
            )

        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.config.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels, hidden_size=self.inner_dim
            )

    def enable_parallel(self, dp_size, sp_size, enable_cp):
        # update cfg parallel
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1

        self.parallel_manager = ParallelManager(dp_size, cp_size, sp_size)

        for _, module in self.named_modules():
            if hasattr(module, "parallel_manager"):
                module.parallel_manager = self.parallel_manager

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        batch_size, c, frame, h, w = hidden_states.shape
        # print('hidden_states.shape', hidden_states.shape)
        frame = frame - use_image_num  # 21-4=17
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        attention_mask_vid, attention_mask_img = None, None
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame+use_image_num, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)
            attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w
            attention_mask_img = attention_mask[:, frame:]  # b, use_image_num, h, w

            if attention_mask_vid.numel() > 0:
                attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(1, self.patch_size_t - 1, 1, 1)
                attention_mask_vid = torch.cat([attention_mask_vid_first_frame, attention_mask_vid], dim=1)
                attention_mask_vid = attention_mask_vid.unsqueeze(1)  # b 1 t h w
                attention_mask_vid = F.max_pool3d(
                    attention_mask_vid,
                    kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
                    stride=(self.patch_size_t, self.patch_size, self.patch_size),
                )
                attention_mask_vid = rearrange(attention_mask_vid, "b 1 t h w -> (b 1) 1 (t h w)")
            if attention_mask_img.numel() > 0:
                attention_mask_img = F.max_pool2d(
                    attention_mask_img,
                    kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size),
                )
                attention_mask_img = rearrange(attention_mask_img, "b i h w -> (b i) 1 (h w)")

            attention_mask_vid = (
                (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None
            )
            attention_mask_img = (
                (1 - attention_mask_img.bool().to(self.dtype)) * -10000.0 if attention_mask_img.numel() > 0 else None
            )

            if frame == 1 and use_image_num == 0:
                attention_mask_img = attention_mask_vid
                attention_mask_vid = None
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, : in_t - use_image_num]  # b, 1, l
            encoder_attention_mask_vid = (
                rearrange(encoder_attention_mask_vid, "b 1 l -> (b 1) 1 l")
                if encoder_attention_mask_vid.numel() > 0
                else None
            )

            encoder_attention_mask_img = encoder_attention_mask[:, in_t - use_image_num :]  # b, use_image_num, l
            encoder_attention_mask_img = (
                rearrange(encoder_attention_mask_img, "b i l -> (b i) 1 l")
                if encoder_attention_mask_img.numel() > 0
                else None
            )

            if frame == 1 and use_image_num == 0:
                encoder_attention_mask_img = encoder_attention_mask_vid
                encoder_attention_mask_vid = None

        if npu_config is not None and attention_mask_vid is not None:
            attention_mask_vid = npu_config.get_attention_mask(attention_mask_vid, attention_mask_vid.shape[-1])
            encoder_attention_mask_vid = npu_config.get_attention_mask(
                encoder_attention_mask_vid, attention_mask_vid.shape[-2]
            )
        if npu_config is not None and attention_mask_img is not None:
            attention_mask_img = npu_config.get_attention_mask(attention_mask_img, attention_mask_img.shape[-1])
            encoder_attention_mask_img = npu_config.get_attention_mask(
                encoder_attention_mask_img, attention_mask_img.shape[-2]
            )

        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        (
            hidden_states_vid,
            hidden_states_img,
            encoder_hidden_states_vid,
            encoder_hidden_states_img,
            timestep_vid,
            timestep_img,
            embedded_timestep_vid,
            embedded_timestep_img,
        ) = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
        )
        # 2. Blocks
        if self.parallel_manager.sp_size > 1:
            if hidden_states_vid is not None:
                hidden_states_vid = split_sequence(
                    hidden_states_vid, dim=1, process_group=self.parallel_manager.sp_group, grad_scale="down"
                )

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if hidden_states_vid is not None:
                    hidden_states_vid = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states_vid,
                        attention_mask_vid,
                        encoder_hidden_states_vid,
                        encoder_attention_mask_vid,
                        timestep_vid,
                        cross_attention_kwargs,
                        class_labels,
                        frame,
                        height,
                        width,
                        **ckpt_kwargs,
                    )
                if hidden_states_img is not None:
                    hidden_states_img = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states_img,
                        attention_mask_img,
                        encoder_hidden_states_img,
                        encoder_attention_mask_img,
                        timestep_img,
                        cross_attention_kwargs,
                        class_labels,
                        1,
                        height,
                        width,
                        **ckpt_kwargs,
                    )
            else:
                if hidden_states_vid is not None:
                    hidden_states_vid = block(
                        hidden_states_vid,
                        attention_mask=attention_mask_vid,
                        encoder_hidden_states=encoder_hidden_states_vid,
                        encoder_attention_mask=encoder_attention_mask_vid,
                        timestep=timestep_vid,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        frame=frame,
                        height=height,
                        width=width,
                        org_timestep=timestep,
                    )
                if hidden_states_img is not None:
                    hidden_states_img = block(
                        hidden_states_img,
                        attention_mask=attention_mask_img,
                        encoder_hidden_states=encoder_hidden_states_img,
                        encoder_attention_mask=encoder_attention_mask_img,
                        timestep=timestep_img,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        frame=1,
                        height=height,
                        width=width,
                        org_timestep=timestep,
                    )

        if self.parallel_manager.sp_size > 1:
            if hidden_states_vid is not None:
                hidden_states_vid = gather_sequence(
                    hidden_states_vid, dim=1, process_group=self.parallel_manager.sp_group, grad_scale="up"
                )

        # 3. Output
        output_vid, output_img = None, None
        if hidden_states_vid is not None:
            output_vid = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_vid,
                timestep=timestep_vid,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid,
                num_frames=frame,
                height=height,
                width=width,
            )  # b c t h w
        if hidden_states_img is not None:
            output_img = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_img,
                timestep=timestep_img,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_img,
                num_frames=1,
                height=height,
                width=width,
            )  # b c 1 h w
            if use_image_num != 0:
                output_img = rearrange(output_img, "(b i) c 1 h w -> b c i h w", i=use_image_num)

        if output_vid is not None and output_img is not None:
            output = torch.cat([output_vid, output_img], dim=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img

        if not return_dict:
            return (output,)

        return VideoSysPipelineOutput(video=output)

    def _operate_on_patched_inputs(
        self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
    ):
        # batch_size = hidden_states.shape[0]
        hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        encoder_hidden_states_vid, encoder_hidden_states_img = None, None

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
            )  # b 6d, b d
            if hidden_states_vid is None:
                timestep_img = timestep
                embedded_timestep_img = embedded_timestep
            else:
                timestep_vid = timestep
                embedded_timestep_vid = embedded_timestep
                if hidden_states_img is not None:
                    timestep_img = repeat(timestep, "b d -> (b i) d", i=use_image_num).contiguous()
                    embedded_timestep_img = repeat(embedded_timestep, "b d -> (b i) d", i=use_image_num).contiguous()

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states
            )  # b, 1+use_image_num, l, d or b, 1, l, d
            if hidden_states_vid is None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states, "b 1 l d -> (b 1) l d")
            else:
                encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], "b 1 l d -> (b 1) l d")
                if hidden_states_img is not None:
                    encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], "b i l d -> (b i) l d")

        return (
            hidden_states_vid,
            hidden_states_img,
            encoder_hidden_states_vid,
            encoder_hidden_states_img,
            timestep_vid,
            timestep_img,
            embedded_timestep_vid,
            embedded_timestep_img,
        )

    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):
        # import ipdb;ipdb.set_trace()
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=self.dtype)
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                num_frames,
                height,
                width,
                self.patch_size_t,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                num_frames * self.patch_size_t,
                height * self.patch_size,
                width * self.patch_size,
            )
        )
        # import ipdb;ipdb.set_trace()
        # if output.shape[2] % 2 == 0:
        #     output = output[:, :, 1:]
        return output


def OpenSoraT2V_S_122(**kwargs):
    return OpenSoraT2V(
        num_layers=28,
        attention_head_dim=96,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1536,
        **kwargs,
    )


def OpenSoraT2V_B_122(**kwargs):
    return OpenSoraT2V(
        num_layers=32,
        attention_head_dim=96,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1920,
        **kwargs,
    )


def OpenSoraT2V_L_122(**kwargs):
    return OpenSoraT2V(
        num_layers=40,
        attention_head_dim=128,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=2048,
        **kwargs,
    )


def OpenSoraT2V_ROPE_L_122(**kwargs):
    return OpenSoraT2V(
        num_layers=32,
        attention_head_dim=96,
        num_attention_heads=24,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=2304,
        **kwargs,
    )


OpenSora_models = {
    "OpenSoraT2V-S/122": OpenSoraT2V_S_122,  #       1.1B
    "OpenSoraT2V-B/122": OpenSoraT2V_B_122,
    "OpenSoraT2V-L/122": OpenSoraT2V_L_122,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V_ROPE_L_122,
}

OpenSora_models_class = {
    "OpenSoraT2V-S/122": OpenSoraT2V,
    "OpenSoraT2V-B/122": OpenSoraT2V,
    "OpenSoraT2V-L/122": OpenSoraT2V,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V,
}
