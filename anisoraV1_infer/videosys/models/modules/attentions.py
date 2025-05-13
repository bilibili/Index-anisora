import inspect
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange
from torch import nn
from torch.amp import autocast

from videosys.core.comm import all_to_all_with_pad, get_pad, set_pad
from videosys.core.pab_mgr import enable_pab, if_broadcast_cross, if_broadcast_spatial, if_broadcast_temporal
from videosys.models.modules.normalization import LlamaRMSNorm, VchitectSpatialNorm
from videosys.utils.logging import logger


class OpenSoraAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OpenSoraMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, enable_flash_attn=False):
        super(OpenSoraMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flash_attn = enable_flash_attn

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        if self.enable_flash_attn:
            x = self.flash_attn_impl(q, k, v, mask, B, N, C)
        else:
            x = self.torch_impl(q, k, v, mask, B, N, C)

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

        attn_mask = torch.zeros(B, 1, N, k.shape[2], dtype=torch.bool, device=q.device)
        for i, m in enumerate(mask):
            attn_mask[i, :, :, :m] = -1e9

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = out.transpose(1, 2).contiguous().view(B, N, C)
        return x


@dataclass
class _SeqLenInfo:
    """
    from xformers

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


class VchitectAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional[AttnProcessor] = None,
        out_dim: int = None,
        context_pre_only: bool = None,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = VchitectSpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.to_q_cross = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_q_temp = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k_temp = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v_temp = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k_temp = None
            self.to_v_temp = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        self.to_out_temporal = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
        nn.init.constant_(self.to_out_temporal.weight, 0)
        nn.init.constant_(self.to_out_temporal.bias, 0)

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
        self.to_add_out_temporal = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
        nn.init.constant_(self.to_add_out_temporal.weight, 0)
        nn.init.constant_(self.to_add_out_temporal.bias, 0)

        self.to_out_context = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
        nn.init.constant_(self.to_out_context.weight, 0)
        nn.init.constant_(self.to_out_context.bias, 0)

        # set attention processor
        self.set_processor(processor)

        # parallel
        self.parallel_manager = None

        # pab
        self.spatial_count = 0
        self.last_spatial = None
        self.temporal_count = 0
        self.last_temporal = None
        self.cross_count = 0
        self.last_cross = None

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        full_seqlen: Optional[int] = None,
        Frame: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            full_seqlen=full_seqlen,
            Frame=Frame,
            timestep=timestep,
            **cross_attention_kwargs,
        )

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse


class VchitectAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @autocast("cuda", enabled=False)
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def spatial_attn(
        self,
        attn,
        hidden_states,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        batch_size,
        head_dim,
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        xq, xk = query.to(value.dtype), key.to(value.dtype)

        query_spatial, key_spatial, value_spatial = xq, xk, value
        query_spatial = query_spatial.transpose(1, 2)
        key_spatial = key_spatial.transpose(1, 2)
        value_spatial = value_spatial.transpose(1, 2)

        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query_spatial, key_spatial, value_spatial, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states

    def temporal_attention(
        self,
        attn,
        hidden_states,
        residual,
        batch_size,
        batchsize,
        Frame,
        head_dim,
        freqs_cis,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
    ):
        query_t = attn.to_q_temp(hidden_states)
        key_t = attn.to_k_temp(hidden_states)
        value_t = attn.to_v_temp(hidden_states)

        query_t = torch.cat([query_t, encoder_hidden_states_query_proj], dim=1)
        key_t = torch.cat([key_t, encoder_hidden_states_key_proj], dim=1)
        value_t = torch.cat([value_t, encoder_hidden_states_value_proj], dim=1)

        query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
        key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
        value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

        query_t, key_t = query_t.to(value_t.dtype), key_t.to(value_t.dtype)

        if attn.parallel_manager.sp_size > 1:
            func = lambda x: self.dynamic_switch(attn, x, batchsize, to_spatial_shard=True)
            query_t, key_t, value_t = map(func, [query_t, key_t, value_t])

        func = lambda x: rearrange(x, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)
        xq_gather_temporal, xk_gather_temporal, xv_gather_temporal = map(func, [query_t, key_t, value_t])

        freqs_cis_temporal = freqs_cis[: xq_gather_temporal.shape[1], :]
        xq_gather_temporal, xk_gather_temporal = self.apply_rotary_emb(
            xq_gather_temporal, xk_gather_temporal, freqs_cis=freqs_cis_temporal
        )

        xq_gather_temporal = xq_gather_temporal.transpose(1, 2)
        xk_gather_temporal = xk_gather_temporal.transpose(1, 2)
        xv_gather_temporal = xv_gather_temporal.transpose(1, 2)

        batch_size_temp = xv_gather_temporal.shape[0]
        hidden_states_temp = F.scaled_dot_product_attention(
            xq_gather_temporal, xk_gather_temporal, xv_gather_temporal, dropout_p=0.0, is_causal=False
        )
        hidden_states_temp = hidden_states_temp.transpose(1, 2).reshape(batch_size_temp, -1, attn.heads * head_dim)
        hidden_states_temp = hidden_states_temp.to(value_t.dtype)
        hidden_states_temp = rearrange(hidden_states_temp, "(B S) T C -> (B T) S C", T=Frame, B=batchsize)
        if attn.parallel_manager.sp_size > 1:
            hidden_states_temp = self.dynamic_switch(attn, hidden_states_temp, batchsize, to_spatial_shard=False)

        hidden_states_temporal, encoder_hidden_states_temporal = (
            hidden_states_temp[:, : residual.shape[1]],
            hidden_states_temp[:, residual.shape[1] :],
        )
        hidden_states_temporal = attn.to_out_temporal(hidden_states_temporal)
        return hidden_states_temporal, encoder_hidden_states_temporal

    def cross_attention(
        self,
        attn,
        hidden_states,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        batch_size,
        head_dim,
        cur_frame,
        batchsize,
    ):
        query_cross = attn.to_q_cross(hidden_states)
        query_cross = torch.cat([query_cross, encoder_hidden_states_query_proj], dim=1)

        key_y = encoder_hidden_states_key_proj[0].unsqueeze(0)
        value_y = encoder_hidden_states_value_proj[0].unsqueeze(0)

        query_y = query_cross.view(batch_size, -1, attn.heads, head_dim)
        key_y = key_y.view(batchsize, -1, attn.heads, head_dim)
        value_y = value_y.view(batchsize, -1, attn.heads, head_dim)

        query_y = rearrange(query_y, "(B T) S H C -> B (S T) H C", T=cur_frame, B=batchsize)

        query_y = query_y.transpose(1, 2)
        key_y = key_y.transpose(1, 2)
        value_y = value_y.transpose(1, 2)

        cross_output = F.scaled_dot_product_attention(query_y, key_y, value_y, dropout_p=0.0, is_causal=False)
        cross_output = cross_output.transpose(1, 2).reshape(batchsize, -1, attn.heads * head_dim)
        cross_output = cross_output.to(query_cross.dtype)

        cross_output = rearrange(cross_output, "B (S T) C -> (B T) S C", T=cur_frame, B=batchsize)
        cross_output = attn.to_out_context(cross_output)
        return cross_output

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        full_seqlen: Optional[int] = None,
        Frame: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        batch_size = encoder_hidden_states.shape[0]
        inner_dim = encoder_hidden_states_key_proj.shape[-1]
        head_dim = inner_dim // attn.heads
        batchsize = full_seqlen // Frame
        # same as Frame if shard, otherwise sharded frame
        cur_frame = batch_size // batchsize

        # temporal attention
        if enable_pab():
            broadcast_temporal, attn.temporal_count = if_broadcast_temporal(int(timestep[0]), attn.temporal_count)
        if enable_pab() and broadcast_temporal:
            hidden_states_temporal, encoder_hidden_states_temporal = attn.last_temporal
        else:
            hidden_states_temporal, encoder_hidden_states_temporal = self.temporal_attention(
                attn,
                hidden_states,
                residual,
                batch_size,
                batchsize,
                Frame,
                head_dim,
                freqs_cis,
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            )
            if enable_pab():
                attn.last_temporal = (hidden_states_temporal, encoder_hidden_states_temporal)

        # cross attn
        if enable_pab():
            broadcast_cross, attn.cross_count = if_broadcast_cross(int(timestep[0]), attn.cross_count)
        if enable_pab() and broadcast_cross:
            cross_output = attn.last_cross
        else:
            cross_output = self.cross_attention(
                attn,
                hidden_states,
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
                batch_size,
                head_dim,
                cur_frame,
                batchsize,
            )
            if enable_pab():
                attn.last_cross = cross_output

        # spatial attn
        if enable_pab():
            broadcast_spatial, attn.spatial_count = if_broadcast_spatial(int(timestep[0]), attn.spatial_count)
        if enable_pab() and broadcast_spatial:
            hidden_states = attn.last_spatial
        else:
            hidden_states = self.spatial_attn(
                attn,
                hidden_states,
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
                batch_size,
                head_dim,
            )
            if enable_pab():
                attn.last_spatial = hidden_states

        # processs attention outputs.
        hidden_states = hidden_states * 1.1 + cross_output
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if cur_frame == 1:
            hidden_states_temporal = hidden_states_temporal * 0
        hidden_states = hidden_states + hidden_states_temporal

        # encoder
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        encoder_hidden_states_temporal = attn.to_add_out_temporal(encoder_hidden_states_temporal)
        if cur_frame == 1:
            encoder_hidden_states_temporal = encoder_hidden_states_temporal * 0
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_temporal

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

    def dynamic_switch(self, attn, x, batchsize, to_spatial_shard: bool):
        if to_spatial_shard:
            scatter_dim, gather_dim = 2, 1
            set_pad("spatial", x.shape[1], attn.parallel_manager.sp_group)
            scatter_pad = get_pad("spatial")
            gather_pad = get_pad("temporal")
        else:
            scatter_dim, gather_dim = 1, 2
            scatter_pad = get_pad("temporal")
            gather_pad = get_pad("spatial")

        x = rearrange(x, "(B T) S ... -> B T S ...", B=batchsize)
        x = all_to_all_with_pad(
            x,
            attn.parallel_manager.sp_group,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            scatter_pad=scatter_pad,
            gather_pad=gather_pad,
        )
        x = rearrange(x, "B T ... -> (B T) ...")
        return x
