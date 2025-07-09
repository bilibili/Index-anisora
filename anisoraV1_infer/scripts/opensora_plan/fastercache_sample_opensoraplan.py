# Adapted from Open-Sora-Plan

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# --------------------------------------------------------

import argparse
import math
import os

import colossalai
import imageio
import torch
from colossalai.cluster import DistCoordinator
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer

from fastercache.dsp.parallel_mgr import set_parallel_manager
from fastercache.models.opensora_plan import LatteT2V, VideoGenPipeline, ae_stride_config, getae_wrapper
from fastercache.utils.utils import merge_args, set_seed

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import (
    LORA_ATTENTION_PROCESSORS,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    AttnProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    SlicedAttnProcessor,
    SpatialNorm,
    XFormersAttnAddedKVProcessor,
    XFormersAttnProcessor,
    logger,
)
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange, repeat
from torch import nn

from fastercache.dsp.comm import (
    all_to_all_with_pad,
    gather_sequence,
    get_spatial_pad,
    get_temporal_pad,
    set_spatial_pad,
    set_temporal_pad,
    split_sequence,
)
from fastercache.dsp.parallel_mgr import enable_sequence_parallel, get_sequence_parallel_group

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c), dtype=torch.uint8)

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r : start_r + h, start_c : start_c + w] = video[i]

    return video_grid



def fastercache_model_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        self.counter+=1
        if self.counter >=40 and self.counter%6!=0:
            single_output = self.fastercache_model_single_forward(hidden_states[1:2],timestep[1:2],encoder_hidden_states[1:2],added_cond_kwargs,class_labels,cross_attention_kwargs,attention_mask,encoder_attention_mask,use_image_num,enable_temporal_attentions,return_dict)[0]
            (bb, cc, tt, hh, ww) = single_output.shape
            cond = rearrange(single_output, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            lf_c, hf_c = fft(cond.float())
            
            if self.counter<=130:
                self.delta_lf = self.delta_lf * 1.1
            elif self.counter>=100:
                self.delta_hf = self.delta_hf * 1.2

            new_hf_uc = self.delta_hf + hf_c
            new_lf_uc = self.delta_lf + lf_c

            combine_uc = new_lf_uc + new_hf_uc
            combined_fft = torch.fft.ifftshift(combine_uc)
            recovered_uncond = torch.fft.ifft2(combined_fft).real
            recovered_uncond = rearrange(recovered_uncond.to(single_output.dtype), "(B T) C H W -> B C T H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            output = torch.cat([recovered_uncond, single_output],dim=0)
        else:
            output = self.fastercache_model_single_forward(hidden_states,timestep,encoder_hidden_states,added_cond_kwargs,class_labels,cross_attention_kwargs,attention_mask,encoder_attention_mask,use_image_num,enable_temporal_attentions,return_dict)[0]
            if self.counter>38:
                (bb, cc, tt, hh, ww) = output.shape
                cond = rearrange(output[1:2], "B C T H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)
                uncond = rearrange(output[0:1], "B C T H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)

                lf_c, hf_c = fft(cond.float())
                lf_uc, hf_uc = fft(uncond.float())

                self.delta_hf = hf_uc - hf_c
                self.delta_lf = lf_uc - lf_c

        return (output,)


import torch.fft
@torch.no_grad()
def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft


def fastercache_model_single_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
        org_timestep = timestep
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_batch_size, frame + use_image_num, h, w), device=hidden_states.device, dtype=hidden_states.dtype
            )
        attention_mask = self.vae_to_diff_mask(attention_mask, use_image_num)
        dtype = attention_mask.dtype
        attention_mask_compress = F.max_pool2d(
            attention_mask.float(), kernel_size=self.compress_kv_factor, stride=self.compress_kv_factor
        )
        attention_mask_compress = attention_mask_compress.to(dtype)

        attention_mask = self.make_attn_mask(attention_mask, frame, hidden_states.dtype)
        attention_mask_compress = self.make_attn_mask(attention_mask_compress, frame, hidden_states.dtype)

        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, "b 1 l -> (b f) 1 l", f=frame).contiguous()
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(
                encoder_attention_mask_video, "b 1 l -> b (1 f) l", f=frame
            ).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, "b n l -> (b n) l").contiguous().unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        # Retrieve lora scale.
        cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hw = (height, width)
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(
                    encoder_hidden_states_video, "b 1 t d -> b (1 f) t d", f=frame
                ).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
                encoder_hidden_states_spatial = rearrange(encoder_hidden_states, "b f t d -> (b f) t d").contiguous()
            else:
                encoder_hidden_states_spatial = repeat(
                    encoder_hidden_states, "b 1 t d -> (b f) t d", f=frame
                ).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
        timestep_temp = repeat(timestep, "b d -> (b p) d", p=num_patches).contiguous()

        pos_hw, pos_t = None, None
        if self.use_rope:
            pos_hw, pos_t = self.make_position(
                input_batch_size, frame, use_image_num, height, width, hidden_states.device
            )

        if enable_sequence_parallel():
            set_temporal_pad(frame + use_image_num)
            set_spatial_pad(num_patches)
            hidden_states = self.split_from_second_dim(hidden_states, input_batch_size)
            encoder_hidden_states_spatial = self.split_from_second_dim(encoder_hidden_states_spatial, input_batch_size)
            timestep_spatial = self.split_from_second_dim(timestep_spatial, input_batch_size)
            attention_mask = self.split_from_second_dim(attention_mask, input_batch_size)
            attention_mask_compress = self.split_from_second_dim(attention_mask_compress, input_batch_size)
            temp_pos_embed = split_sequence(
                self.temp_pos_embed, get_sequence_parallel_group(), dim=1, grad_scale="down", pad=get_temporal_pad()
            )
        else:
            temp_pos_embed = self.temp_pos_embed

        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    attention_mask_compress if i >= self.num_layers // 2 else attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw,
                    pos_hw,
                    hw,
                    use_reentrant=False,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0:  # image-video joitn training
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states_video = hidden_states_video + temp_pos_embed

                        hidden_states_video = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t,
                            pos_t,
                            (frame,),
                            use_reentrant=False,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + temp_pos_embed

                        hidden_states = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t,
                            pos_t,
                            (frame,),
                            use_reentrant=False,
                        )

                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    attention_mask_compress if i >= self.num_layers // 2 else attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw,
                    pos_hw,
                    hw,
                    org_timestep,
                    self.counter,
                )

                if enable_temporal_attentions:
                    # b c f h w, f = 16 + 4
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0 and self.training:
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        #     hidden_states_video = hidden_states_video + temp_pos_embed

                        hidden_states_video = temp_block(
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t,
                            pos_t,
                            (frame,),
                            org_timestep,
                            self.counter,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + temp_pos_embed

                        hidden_states = temp_block(
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t,
                            pos_t,
                            (frame,),
                            org_timestep,
                            self.counter,
                        )

                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

        if enable_sequence_parallel():
            hidden_states = self.gather_from_second_dim(hidden_states, input_batch_size)

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                embedded_timestep = repeat(embedded_timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, "(b f) c h w -> b c f h w", b=input_batch_size).contiguous()

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


def fastercache_tmp_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        position_q: Optional[torch.LongTensor] = None,
        position_k: Optional[torch.LongTensor] = None,
        frame: int = None,
        org_timestep: Optional[torch.LongTensor] = None,
        counter=None,
    ) -> torch.FloatTensor:
    if True:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
 
        if counter>=45 and counter%3!=0 and self.tacache[1].shape[0]>=hidden_states.shape[0]:
            # attn_output = self.last_out
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            attn_output = self.tacache[1][:hidden_states.shape[0]] + (self.tacache[1][:hidden_states.shape[0]] - self.tacache[0][:hidden_states.shape[0]]) * 0.3
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output
        else:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            if enable_sequence_parallel():
                norm_hidden_states = self.dynamic_switch(norm_hidden_states, to_spatial_shard=True)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                position_q=position_q,
                position_k=position_k,
                last_shape=frame,
                **cross_attention_kwargs,
            )

            if counter==40:
                self.tacache = [attn_output, attn_output]
            elif counter>40:
                self.tacache = [self.tacache[-1],attn_output]

            if enable_sequence_parallel():
                attn_output = self.dynamic_switch(attn_output, to_spatial_shard=False)

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output


        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])


        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            # norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = self.norm3(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


def fastercache_spa_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        position_q: Optional[torch.LongTensor] = None,
        position_k: Optional[torch.LongTensor] = None,
        hw: Tuple[int, int] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        counter=None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if counter>=45 and counter%3!=0 and self.sacache[1].shape[0]>=hidden_states.shape[0]:
            # attn_output = self.spatial_last
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

            attn_output = self.sacache[1][:hidden_states.shape[0]] + (self.sacache[1][:hidden_states.shape[0]] - self.sacache[0][:hidden_states.shape[0]]) * 0.3

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

        else:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                position_q=position_q,
                position_k=position_k,
                last_shape=hw,
                **cross_attention_kwargs,
            )

            if counter==42:
                self.sacache = [attn_output, attn_output]
            elif counter>42:
                self.sacache = [self.sacache[-1],attn_output]


            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if True:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_q=None,  # cross attn do not need relative position
                    position_k=None,
                    last_shape=None,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


def main(args):
    set_seed(42)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    set_parallel_manager(1, coordinator.world_size)
    device = f"cuda:{torch.cuda.current_device()}"

    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir).to(
        device, dtype=torch.float16
    )
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # Load model:
    transformer_model = LatteT2V.from_pretrained(
        args.model_path, subfolder=args.version, cache_dir=args.cache_dir, torch_dtype=torch.float16
    ).to(device)

    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_name, cache_dir=args.cache_dir, torch_dtype=torch.float16
    ).to(device)

    if args.force_images:
        ext = "jpg"
    else:
        ext = "mp4"

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    for _name, _module in transformer_model.named_modules():
        # print(_name,_module)
        if _module.__class__.__name__=='LatteT2V':
            _module.__class__.forward = fastercache_model_forward
            _module.__class__.fastercache_model_single_forward = fastercache_model_single_forward
        if _module.__class__.__name__=='BasicTransformerBlock_':
            _module.__class__.forward = fastercache_tmp_forward
        if _module.__class__.__name__=='BasicTransformerBlock':
            _module.__class__.forward = fastercache_spa_forward

    if args.sample_method == "DDIM":  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == "DDPM":  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler()
    elif args.sample_method == "HeunDiscrete":  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == "KDPM2AncestralDiscrete":  #########
        scheduler = KDPM2AncestralDiscreteScheduler()

    os.makedirs(args.save_img_path, exist_ok=True)
    g = torch.Generator()
    g.manual_seed(101)
    videogen_pipeline = VideoGenPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device=device)

    video_grids = []

    def load_prompts(prompt_path, start_idx=None, end_idx=None):
        with open(prompt_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        prompts = prompts[start_idx:end_idx]
        return prompts
    prompts = None
    if prompts is None:
        assert args.prompt_path is not None
        prompts = load_prompts(args.prompt_path)
    
    latents = videogen_pipeline.prepare_latents(
            1 * 1,
            videogen_pipeline.transformer.config.in_channels,
            args.num_frames,
            args.height,
            args.width,
            torch.float16,
            videogen_pipeline.text_encoder.device,
            generator=g,
            latents=None,
    )

    for idx, prompt in enumerate(prompts):
        print("Processing the ({}) prompt".format(prompt))

        transformer_model.counter = 0

        videos = videogen_pipeline(
            prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=not args.force_images,
            num_images_per_prompt=1,
            mask_feature=True,
            generator=g,
            latents=latents,
            output_type="latents",
        ).video

        video = videogen_pipeline.decode_latents(videos)
        video = video[:, :args.num_frames, :args.height, :args.width]
        videos = video.detach().cpu()

        if coordinator.is_master():
            print('saving to....:',args.save_img_path + prompt[:30].replace(" ", "_") + ".mp4")
            imageio.mimwrite(
                        args.save_img_path + str(idx)+'_'+ prompt[:30].replace(" ", "_") + ".mp4",
                        videos[0],
                        fps=args.fps,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.0.0")
    parser.add_argument("--version", type=str, default=None, choices=[None, "65x512x512", "221x512x512", "513x512x512"])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--force_images", action="store_true")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument('--prompt_path',type=str, default="")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
