from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from ...modules.diffusionmodules.sampling import VideoDDIMSampler, VPSDEDPMPP2MSampler
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
#import rearrange
from einops import rearrange
import random
from sat import mpu
class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
        beta_dpo = 5000,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.beta_dpo = beta_dpo

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, ref_network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        w_score,l_score = batch['w_score']/100.0,batch['l_score']/100.0
        input_w,input_l = input.chunk(2,dim=2)  
        alphas_cumprod_sqrt, idx = self.sigma_sampler(input_w.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input_w.device)
        idx = idx.to(input_w.device)
        noise = torch.randn_like(input_w)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs['idx'] = idx

        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
        noised_input_ = [input_w.float() * append_dims(alphas_cumprod_sqrt, input_w.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input_w.ndim), 
                        input_l.float() * append_dims(alphas_cumprod_sqrt, input_l.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input_l.ndim)]
        concat_images = torch.cat(batch["concat_images"].chunk(2,dim=2))  
        noised_input = torch.cat(noised_input_)
        input = torch.cat([input_w,input_l])
        cond['crossattn'] = cond['crossattn'].repeat(2, 1, 1)

        if "concat_images" in batch.keys():
            additional_model_inputs["concat_images"] = concat_images
        #print(additional_model_inputs.keys()) #dict_keys([])
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        model_loss = self.get_loss(model_output, input, w)

        model_loss_w, model_loss_l = model_loss.chunk(2)
        raw_model_loss = 0.5 * (model_loss_w.mean() + model_loss_l.mean())  
        model_diff = model_loss_w - model_loss_l

        with torch.no_grad():
            ref_output = denoiser(
                ref_network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
            ).detach()
            ref_loss = self.get_loss(ref_output, input, w)
            ref_loss_w, ref_loss_l = ref_loss.chunk(2)
            raw_ref_loss = ref_loss.mean()
            ref_diff = ref_loss_w - ref_loss_l

        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)

        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        loss = -1 * F.logsigmoid(inside_term).mean()
        loss = (2**w_score - 2**l_score)*loss
        #loss = (3**w_score - 3**l_score)*loss
        #loss = (1.0 - 2**l_score/2**w_score)*loss
        return loss

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


def get_3d_position_ids(frame_len, h, w):
    i = torch.arange(frame_len).view(frame_len, 1, 1).expand(frame_len, h, w)
    j = torch.arange(h).view(1, h, 1).expand(frame_len, h, w)
    k = torch.arange(w).view(1, 1, w).expand(frame_len, h, w)
    position_ids = torch.stack([i, j, k], dim=-1).reshape(-1, 3)
    return position_ids