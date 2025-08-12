# Adapted from Latte

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte: https://github.com/Vchitect/Latte
# --------------------------------------------------------
import argparse
import os

import colossalai
import imageio
import torch
from colossalai.cluster import DistCoordinator
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
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
from fastercache.models.latte import LattePipeline, LatteT2V
from fastercache.utils.utils import merge_args, set_seed

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    ImagePositionalEmbeddings,
    PatchEmbed,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    PixArtAlphaTextProjection,
    SinusoidalPositionalEmbedding,
    get_1d_sincos_pos_embed_from_grid,
)
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
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
from einops import rearrange

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
        if self.counter >=18 and self.counter%5!=0:
            single_output = self.fastercache_model_single_forward(hidden_states[1:2],timestep[1:2],encoder_hidden_states[1:2],added_cond_kwargs,class_labels,cross_attention_kwargs,attention_mask,encoder_attention_mask,use_image_num,enable_temporal_attentions,return_dict)[0]
            (bb, cc, tt, hh, ww) = single_output.shape
            cond = rearrange(single_output, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            lf_c, hf_c = fft(cond.float())

            if self.counter<45:
                self.delta_lf = self.delta_lf * 1.1
            elif self.counter>35:
                self.delta_hf = self.delta_hf * 1.1

            new_hf_uc = self.delta_hf + hf_c
            new_lf_uc = self.delta_lf + lf_c

            combine_uc = new_lf_uc + new_hf_uc
            combined_fft = torch.fft.ifftshift(combine_uc)
            recovered_uncond = torch.fft.ifft2(combined_fft).real
            recovered_uncond = rearrange(recovered_uncond.to(single_output.dtype), "(B T) C H W -> B C T H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            output = torch.cat([recovered_uncond, single_output],dim=0)
        else:
            output = self.fastercache_model_single_forward(hidden_states,timestep,encoder_hidden_states,added_cond_kwargs,class_labels,cross_attention_kwargs,attention_mask,encoder_attention_mask,use_image_num,enable_temporal_attentions,return_dict)[0]
            
            if self.counter >= 16:
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
        frame = frame - use_image_num
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
        org_timestep = timestep

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, "b 1 l -> (b f) 1 l", f=frame).contiguous()
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(
                encoder_attention_mask_video, "b 1 l -> b (1 f) l", f=frame
            ).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, "b n l -> (b n) l").contiguous().unsqueeze(1)

        # Retrieve lora scale.
        cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

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
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152

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
                    encoder_hidden_states, "b t d -> (b f) t d", f=frame
                ).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
        timestep_temp = repeat(timestep, "b d -> (b p) d", p=num_patches).contiguous()

        if enable_sequence_parallel():
            set_temporal_pad(frame + use_image_num)
            set_spatial_pad(num_patches)
            hidden_states = self.split_from_second_dim(hidden_states, input_batch_size)
            encoder_hidden_states_spatial = self.split_from_second_dim(encoder_hidden_states_spatial, input_batch_size)
            timestep_spatial = self.split_from_second_dim(timestep_spatial, input_batch_size)
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
                    attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0:  # image-video joitn training
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

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
                            use_reentrant=False,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
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
                            use_reentrant=False,
                        )

                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    None,
                    org_timestep,
                    self.counter
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                    if use_image_num != 0 and self.training:
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        hidden_states_video = temp_block(
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            org_timestep,
                            self.counter,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(
                            hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                        ).contiguous()

                    else:
                        if i == 0 and frame > 1:
                            hidden_states = hidden_states + temp_pos_embed

                        hidden_states = temp_block(
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            org_timestep,
                            self.counter
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

def fastercache_spa_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        org_timestep: Optional[torch.LongTensor] = None,
        counter=None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if counter>=16 and counter%2==0 and self.sacache[1].shape[0]>=hidden_states.shape[0]:
            # attn_output = self.spatial_last
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

            attn_output = self.sacache[1][:hidden_states.shape[0]] + (self.sacache[1][:hidden_states.shape[0]] - self.sacache[0][:hidden_states.shape[0]]) * 0.5

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

        else:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
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
                **cross_attention_kwargs,
            )
            if counter==13:
                self.sacache = [attn_output, attn_output]
            elif counter>13:
                self.sacache = [self.sacache[-1],attn_output]


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
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

def fastercache_tmp_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
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

        if counter>=16 and counter%2==0 and self.tacache[1].shape[0]>=hidden_states.shape[0]:
            # attn_output = self.last_out
            assert self.use_ada_layer_norm_single
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            attn_output = self.tacache[1][:hidden_states.shape[0]] + (self.tacache[1][:hidden_states.shape[0]] - self.tacache[0][:hidden_states.shape[0]]) * 0.5
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
            elif self.use_ada_layer_norm_single:  # go here
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                # norm_hidden_states = norm_hidden_states.squeeze(1)
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
                **cross_attention_kwargs,
            )

            if counter==13:
                self.tacache = [attn_output, attn_output]
            elif counter>13:
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


times_list = []

def main(args):
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    set_parallel_manager(1, coordinator.world_size)
    device = f"cuda:{torch.cuda.current_device()}"

    args.pretrained_model_path='../OpenDiT/Latte-1'
    transformer_model = LatteT2V.from_pretrained(
        args.pretrained_model_path, subfolder="transformer", video_length=args.video_length
    ).to(device, dtype=torch.float16)

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(
            device
        )
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

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

    if args.sample_method == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "HeunDiscrete":
        scheduler = HeunDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "KDPM2AncestralDiscrete":
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )

    g = torch.Generator()
    g.manual_seed(101)
    videogen_pipeline = LattePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device)

    os.makedirs(args.save_img_path, exist_ok=True)

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
            args.video_length,
            args.image_size[0],
            args.image_size[1],
            torch.float16,
            videogen_pipeline._execution_device,
            generator=g,
            latents=None,
    )

    # video_grids = []
    for num_prompt, prompt in enumerate(prompts):

        print("Processing the ({}) prompt".format(prompt))
        transformer_model.counter = 0


        videos = videogen_pipeline(
            prompt,
            video_length=args.video_length,
            height=args.image_size[0],
            width=args.image_size[1],
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=args.enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
            generator=g,
            latents=latents,
            output_type="latents",
        ).video
        
        if videos.shape[2] == 1:  # image
                video = videogen_pipeline.decode_latents_image(videos)
        else:  # video
            if args.enable_vae_temporal_decoder:
                video = videogen_pipeline.decode_latents_with_temporal_decoder(videos)
            else:
                video = videogen_pipeline.decode_latents(videos)
        videos = video.detach().cpu()

        if coordinator.is_master():
            if videos.shape[1] == 1:
                save_image(videos[0][0], args.save_img_path + prompt[:30].replace(" ", "_") + ".png")
            else:
                imageio.mimwrite(
                    args.save_img_path + str(num_prompt)+'_'+prompt[:30].replace(" ", "_") + "_%04d" % args.run_time + ".mp4",
                    videos[0],
                    fps=8,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_img_path", type=str, default="./samples/latte/")
    parser.add_argument("--pretrained_model_path", type=str, default="maxin-cn/Latte-1")
    parser.add_argument("--model", type=str, default="LatteT2V")
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--image_size", nargs="+")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="learned_range")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DDIM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--enable_temporal_attentions", action="store_true")
    parser.add_argument("--enable_vae_temporal_decoder", action="store_true")
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument('--prompt_path',type=str, default="")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)
    print(args.save_img_path)

    main(args)
