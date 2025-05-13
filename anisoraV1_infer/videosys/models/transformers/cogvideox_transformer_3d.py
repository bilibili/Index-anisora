# Adapted from CogVideo

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# CogVideo: https://github.com/THUDM/CogVideo
# diffusers: https://github.com/huggingface/diffusers
# --------------------------------------------------------

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

from videosys.core.comm import all_to_all_comm, gather_sequence, get_pad, set_pad, split_sequence
from videosys.core.pab_mgr import enable_pab, if_broadcast_spatial
from videosys.core.parallel_mgr import ParallelManager
from videosys.models.modules.embeddings import apply_rotary_emb
from videosys.utils.utils import batch_func

from ..modules.embeddings import CogVideoXPatchEmbed
from ..modules.normalization import AdaLayerNorm, CogVideoXLayerNormZero


class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _remove_extra_encoder(self, hidden_states, text_seq_length, attn):
        # current layout is [text, 1/n seq, text, 1/n seq, ...]
        # we want to remove the all the text info [text, seq]
        sp_size = attn.parallel_manager.sp_size
        split_seq = hidden_states.split(hidden_states.size(2) // sp_size, dim=2)
        encoder_hidden_states = split_seq[0][:, :, :text_seq_length]
        new_seq = [encoder_hidden_states]
        for i in range(sp_size):
            new_seq.append(split_seq[i][:, :, text_seq_length:])
        hidden_states = torch.cat(new_seq, dim=2)

        # remove padding added when all2all
        # if pad is removed earlier than this
        # the split size will be wrong
        pad = get_pad("pad")
        if pad > 0:
            hidden_states = hidden_states.narrow(2, 0, hidden_states.size(2) - pad)
        return hidden_states

    def _add_extra_encoder(self, hidden_states, text_seq_length, attn):
        # add padding for split and later all2all
        # if pad is removed later than this
        # the split size will be wrong
        pad = get_pad("pad")
        if pad > 0:
            pad_shape = list(hidden_states.shape)
            pad_shape[1] = pad
            pad_tensor = torch.zeros(pad_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, pad_tensor], dim=1)

        # current layout is [text, seq]
        # we want to add the extra encoder info [text, 1/n seq, text, 1/n seq, ...]
        sp_size = attn.parallel_manager.sp_size
        encoder = hidden_states[:, :text_seq_length]
        seq = hidden_states[:, text_seq_length:]
        seq = seq.split(seq.size(1) // sp_size, dim=1)
        new_seq = []
        for i in range(sp_size):
            new_seq.append(encoder)
            new_seq.append(seq[i])
        hidden_states = torch.cat(new_seq, dim=1)
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # print(333333,query.shape,key.shape,value.shape)
        if attn.parallel_manager.sp_size > 1:
            # assert (
            #     attn.heads % attn.parallel_manager.sp_size == 0
            # ), f"Number of heads {attn.heads} must be divisible by sequence parallel size {attn.parallel_manager.sp_size}"
            attn_heads = attn.heads // attn.parallel_manager.sp_size
            # normally we operate pad for every all2all. but for more convient implementation
            # we move pad operation to encoder add and remove in cogvideo
            query, key, value = map(
                lambda x: all_to_all_comm(x, attn.parallel_manager.sp_group, scatter_dim=2, gather_dim=1),
                [query, key, value],
            )
        else:
            attn_heads = attn.heads


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_heads
        # print(444444,query.shape,key.shape,value.shape,attn_heads,head_dim)

        query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attn.parallel_manager.sp_size > 1:
            # remove extra encoder for attention
            query, key, value = map(
                lambda x: self._remove_extra_encoder(x, text_seq_length, attn),
                [query, key, value],
            )

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            emb_len = image_rotary_emb[0].shape[0]
            query[:, :, text_seq_length : emb_len + text_seq_length] = apply_rotary_emb(
                query[:, :, text_seq_length : emb_len + text_seq_length], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length : emb_len + text_seq_length] = apply_rotary_emb(
                    key[:, :, text_seq_length : emb_len + text_seq_length], image_rotary_emb
                )

        # print(5555555,query.shape,key.shape,value.shape)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # print(666666,hidden_states.shape)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn_heads * head_dim)
        # print(662323266,hidden_states.shape)

        if attn.parallel_manager.sp_size > 1:
            # add extra encoder for all_to_all
            hidden_states = self._add_extra_encoder(hidden_states, text_seq_length, attn)
            hidden_states = all_to_all_comm(hidden_states, attn.parallel_manager.sp_group, scatter_dim=1, gather_dim=2)

        # print(777777777,hidden_states.shape)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        block_idx: int = 0,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # parallel
        self.attn1.parallel_manager = None

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # pab
        self.attn_count = 0
        self.last_attn = None
        self.block_idx = block_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep=None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        if enable_pab():
            broadcast_attn, self.attn_count = if_broadcast_spatial(int(timestep[0]), self.attn_count)
        if enable_pab() and broadcast_attn:
            attn_hidden_states, attn_encoder_hidden_states = self.last_attn
        else:
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
            if enable_pab():
                self.last_attn = (attn_hidden_states, attn_encoder_hidden_states)

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1
        self.num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(patch_size, in_channels, inner_dim, text_embed_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. 3D positional embeddings
        spatial_pos_embedding = get_3d_sincos_pos_embed(
            inner_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        )
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1)
        pos_embedding = torch.zeros(1, max_text_seq_length + self.num_patches, inner_dim, requires_grad=False)
        pos_embedding.data[:, max_text_seq_length:].copy_(spatial_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 3. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 4. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        # parallel
        self.parallel_manager = None

    def enable_parallel(self, dp_size, sp_size, enable_cp):
        # update cfg parallel
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1

        self.parallel_manager: ParallelManager = ParallelManager(dp_size, cp_size, sp_size)

        for _, module in self.named_modules():
            if hasattr(module, "parallel_manager"):
                module.parallel_manager = self.parallel_manager

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        if self.parallel_manager.cp_size > 1:
            (
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                image_rotary_emb,
            ) = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                image_rotary_emb,
            )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)#

        # 3. Position embedding
        text_seq_length = encoder_hidden_states.shape[1]
        if not self.config.use_rotary_positional_embeddings:
            seq_length = height * width * num_frames // (self.config.patch_size**2)

            pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
            hidden_states = hidden_states + pos_embeds
            hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.parallel_manager.sp_size > 1:
            # print(1111111,hidden_states.shape)
            set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
            hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
            # print(22222222,get_pad("pad"),hidden_states.shape)
            #22222222 0 torch.Size([2, 17550,3072])#5Bsp1   #208.5s #pad0
            #22222222 0 torch.Size([2, 8775, 3072])#5Bsp2   #119.4s #pad0
            #22222222 2 torch.Size([2, 4388, 3072])#5Bsp4   #65.8s  #pad2
            #22222222 2 torch.Size([2, 2194, 3072])#5Bsp8   #43.5s  #pad2
        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    timestep=timesteps if enable_pab() else None,
                )

        if self.parallel_manager.sp_size > 1:
            hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if self.parallel_manager.cp_size > 1:
            output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
