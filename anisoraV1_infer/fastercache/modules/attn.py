from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from fastercache.modules.layers import LlamaRMSNorm


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flashattn: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flashattn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))
            x = F.scaled_dot_product_attention(
                q, k, v, scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0
            )

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, enable_flashattn=False):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.enable_flashattn = enable_flashattn

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_out = None
        self.count = 0

    def forward(self, x, cond, mask=None, timestep=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        x = self.flash_attn_impl(q, k, v, mask, B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flash_attn_impl(self, q, k, v, mask, B, N, C):
        from flash_attn import flash_attn_varlen_func

        q_seqinfo = _SeqLenInfo.from_seqlens([N] * B)
        k_seqinfo = _SeqLenInfo.from_seqlens(mask)

        x = flash_attn_varlen_func(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_heads, self.head_dim),
            v.view(-1, self.num_heads, self.head_dim),
            cu_seqlens_q=q_seqinfo.seqstart.cuda(),
            cu_seqlens_k=k_seqinfo.seqstart.cuda(),
            max_seqlen_q=q_seqinfo.max_seqlen,
            max_seqlen_k=k_seqinfo.max_seqlen,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.view(B, N, C)
        return x

    def torch_impl(self, q, k, v, mask, B, N, C):
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.zeros(B, N, k.shape[2], dtype=torch.float32, device=q.device)
        for i, m in enumerate(mask):
            attn_mask[i, :, m:] = -1e8

        scale = 1 / q.shape[-1] ** 0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.to(torch.float32)
        if mask is not None:
            attn = attn + attn_mask.unsqueeze(1)
        attn = attn.softmax(-1)
        attn = attn.to(v.dtype)
        out = attn @ v

        x = out.transpose(1, 2).contiguous().view(B, N, C)
        return x


@dataclass
class _SeqLenInfo:
    """
    copied from xformers

    (Internal) Represents the division of a dimension into blocks.
    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """

    seqstart: torch.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def to(self, device: torch.device) -> None:
        self.seqstart = self.seqstart.to(device, non_blocking=True)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = torch.tensor(seqstart_py, dtype=torch.int32)
        return cls(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )

    def split(self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
        if self.seqstart_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `torch.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.seqstart_py[-1]}\n"
                f" seqstart: {self.seqstart_py}"
            )
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(self.seqstart_py[it + batch_size] - self.seqstart_py[it])
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]]) for bs, tensor in zip(batch_sizes, x.split(split_chunks, dim=1))
        ]
