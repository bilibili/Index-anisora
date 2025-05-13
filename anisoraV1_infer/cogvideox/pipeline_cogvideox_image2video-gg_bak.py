# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import PIL
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from .embeddings import get_3d_rotary_pos_embed
from .pipeline_output import CogVideoXPipelineOutput
from .cogvideox_transformer_3d import CogVideoXTransformer3DModel
import pdb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import CogVideoXImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> video = pipe(image, prompt, use_dynamic_cfg=True)
        >>> export_to_video(video.frames[0], "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

import torch.fft
from einops import rearrange
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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

import inspect
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5Tokenizer

from videosys.core.pab_mgr import PABConfig, set_pab_manager, update_steps
from videosys.core.pipeline import VideoSysPipeline, VideoSysPipelineOutput
from videosys.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
# from videosys.models.modules.embeddings import get_3d_rotary_pos_embed
# from videosys.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from videosys.schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
from videosys.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
from videosys.utils.logging import logger
from videosys.utils.utils import save_video, set_seed
class CogVideoXPABConfig(PABConfig):
    def __init__(
        self,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [100, 850],
        spatial_range: int = 2,
    ):
        super().__init__(
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
        )


class CogVideoXConfig:
    """
    This config is to instantiate a `CogVideoXPipeline` class for video generation.

    To be specific, this config will be passed to engine by `VideoSysEngine(config)`.
    In the engine, it will be used to instantiate the corresponding pipeline class.
    And the engine will call the `generate` function of the pipeline to generate the video.
    If you want to explore the detail of generation, please refer to the pipeline class below.

    Args:
        model_path (str):
            A path to the pretrained pipeline. Defaults to "THUDM/CogVideoX-2b".
        num_gpus (int):
            The number of GPUs to use. Defaults to 1.
        cpu_offload (bool):
            Whether to enable CPU offload. Defaults to False.
        vae_tiling (bool):
            Whether to enable tiling for the VAE. Defaults to True.
        enable_pab (bool):
            Whether to enable Pyramid Attention Broadcast. Defaults to False.
        pab_config (CogVideoXPABConfig):
            The configuration for Pyramid Attention Broadcast. Defaults to `CogVideoXPABConfig()`.

    Examples:
        ```python
        from videosys import CogVideoXConfig, VideoSysEngine

        # models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        # change num_gpus for multi-gpu inference
        config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
        engine = VideoSysEngine(config)

        prompt = "Sunset over the sea."
        # num frames should be <= 49. resolution is fixed to 720p.
        video = engine.generate(
            prompt=prompt,
            guidance_scale=6,
            num_inference_steps=50,
            num_frames=49,
        ).video[0]
        engine.save_video(video, f"./outputs/{prompt}.mp4")
        ```
    """

    def __init__(
        self,
        model_path: str = "THUDM/CogVideoX-2b",
        # ======= distributed ========
        num_gpus: int = 1,
        # ======= memory =======f
        cpu_offload: bool = False,
        vae_tiling: bool = True,
        # ======= pab ========
        enable_pab: bool = False,
        pab_config=CogVideoXPABConfig(),
    ):
        self.model_path = model_path
        self.pipeline_cls = CogVideoXPipeline
        # ======= distributed ========
        self.num_gpus = num_gpus
        # ======= memory ========
        self.cpu_offload = cpu_offload
        self.vae_tiling = vae_tiling
        # ======= pab ========
        self.enable_pab = enable_pab
        self.pab_config = pab_config

# class CogVideoXImageToVideoPipeline(DiffusionPipeline):
class CogVideoXImageToVideoPipeline(VideoSysPipeline):
    r"""
    Pipeline for image-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        config: CogVideoXConfig,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        vae: Optional[AutoencoderKLCogVideoX] = None,
        transformer: Optional[CogVideoXTransformer3DModel] = None,
        scheduler: Optional[CogVideoXDDIMScheduler] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # print(-1,device,config.cpu_offload)#cuda#False
        self._config = config
        self._device = device
        if config.model_path == "THUDM/CogVideoX-2b":
            dtype = torch.float16
        self._dtype = dtype


        if transformer is None:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                config.model_path, subfolder="transformer", torch_dtype=self._dtype
            )
        if vae is None:
            vae = AutoencoderKLCogVideoX.from_pretrained(config.model_path, subfolder="vae", torch_dtype=self._dtype)
        if tokenizer is None:
            tokenizer = T5Tokenizer.from_pretrained(config.model_path, subfolder="tokenizer")
        if text_encoder is None:
            text_encoder = T5EncoderModel.from_pretrained(
                config.model_path, subfolder="text_encoder", torch_dtype=self._dtype
            )
        if scheduler is None:
            scheduler = CogVideoXDDIMScheduler.from_pretrained(
                config.model_path,
                subfolder="scheduler",
            )

        # self.transformer=transformer
        # self.tokenizer=tokenizer
        # self.text_encoder=text_encoder
        # self.vae=vae
        # self.scheduler=scheduler

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # cpu offload
        # if config.cpu_offload:
        #     self.enable_model_cpu_offload()
        # else:
        #     self.set_eval_and_device(self._device, text_encoder, vae, transformer)

        # self.set_eval_and_device(self._device, text_encoder, vae, transformer)
        # if config.cpu_offload:
        #     self.enable_model_cpu_offload()

        # vae tiling
        if config.vae_tiling:
            vae.enable_tiling()

        # pab
        if config.enable_pab:
            set_pab_manager(config.pab_config)


        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # parallel
        self._set_parallel()

    def _set_seed(self, seed):
        if dist.get_world_size() == 1:
            set_seed(seed)
        else:
            set_seed(seed, None)

    def _set_parallel(
            self, dp_size: Optional[int] = None, sp_size: Optional[int] = None, enable_cp: Optional[bool] = False
    ):
        # init sequence parallel
        if sp_size is None:
            sp_size = dist.get_world_size()
            dp_size = 1
        else:
            assert (
                    dist.get_world_size() % sp_size == 0
            ), f"world_size {dist.get_world_size()} must be divisible by sp_size"
            dp_size = dist.get_world_size() // sp_size

        # transformer parallel
        self.transformer.enable_parallel(dp_size, sp_size, enable_cp)

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # print(0,device,self._execution_device)#0 cuda cpu
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        # print(1,device)#cuda
        # print(2,text_input_ids.device)#cpu
        # print(3,self.text_encoder.device)#cuda:0
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        self.text_encoder=self.text_encoder.to(device)
        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        self.text_encoder=self.text_encoder.cpu()
        torch.cuda.empty_cache()
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        image = image.unsqueeze(2)  # [B, C, F, H, W]

        self.vae=self.vae.to(image.device)
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]
        self.vae=self.vae.cpu()
        torch.cuda.empty_cache()

        image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents = self.vae_scaling_factor_image * image_latents

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    def prepare_latents_2img(
            self,
            image: torch.Tensor,
            image_last: torch.Tensor,
            batch_size: int = 1,
            num_channels_latents: int = 16,
            num_frames: int = 13,
            height: int = 60,
            width: int = 90,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        image = image.unsqueeze(2)  # [B, C, F, H, W]
        image_last = image_last.unsqueeze(2)

        self.vae = self.vae.to(image.device)
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
            image_latents_last = [
                retrieve_latents(self.vae.encode(image_last[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]

        else:
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]
            image_latents_last = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image_last]
        self.vae = self.vae.cpu()
        torch.cuda.empty_cache()

        image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents = self.vae_scaling_factor_image * image_latents

        image_latents_last = torch.cat(image_latents_last, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents_last = self.vae_scaling_factor_image * image_latents_last

        padding_shape1 = (
            batch_size,
            num_frames - 2,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding1 = torch.zeros(padding_shape1, device=device, dtype=dtype)

        image_latents = torch.cat([image_latents, latent_padding1, image_latents_last], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    def prepare_latents_3img(
            self,
            image: torch.Tensor,
            image_mid: torch.Tensor,
            image_last: torch.Tensor,
            batch_size: int = 1,
            num_channels_latents: int = 16,
            num_frames: int = 13,
            height: int = 60,
            width: int = 90,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # image = image.unsqueeze(2)  # [B, C, F, H, W]
        # image_mid = image_mid.unsqueeze(2)  # [B, C, F, H, W]
        # image_last = image_last.unsqueeze(2)
        #
        self.vae = self.vae.to(image.device)
        # if isinstance(generator, list):
        #     image_latents = [
        #         retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
        #     ]
        #     image_latents_mid = [
        #         retrieve_latents(self.vae.encode(image_mid[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
        #     ]
        #     image_latents_last = [
        #         retrieve_latents(self.vae.encode(image_last[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
        #     ]
        #
        # else:
        #     image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]
        #     image_latents_mid = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image_mid]
        #     image_latents_last = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image_last]

        imgs=torch.cat([image,image_mid,image_last],0)
        imgs=imgs.unsqueeze(2)
        img_latents = retrieve_latents(self.vae.encode(imgs), generator)

        self.vae = self.vae.cpu()
        torch.cuda.empty_cache()

        # image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        # image_latents = self.vae_scaling_factor_image * image_latents
        #
        # image_latents_mid = torch.cat(image_latents_mid, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        # image_latents_mid = self.vae_scaling_factor_image * image_latents_mid
        #
        # image_latents_last = torch.cat(image_latents_last, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        # image_latents_last = self.vae_scaling_factor_image * image_latents_last

        img_latents*=self.vae_scaling_factor_image
        img_latents=img_latents.to(dtype).permute(0, 2, 1, 3, 4).unsqueeze(1)
        image_latents,image_latents_mid,image_latents_last=img_latents

        padding_shape1 = (
            batch_size,
            (num_frames - 3) // 2,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        padding_shape2 = (
            batch_size,
            (num_frames - 3) - ((num_frames - 3) // 2),
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding1 = torch.zeros(padding_shape1, device=device, dtype=dtype)
        latent_padding2 = torch.zeros(padding_shape2, device=device, dtype=dtype)

        image_latents = torch.cat([image_latents, latent_padding1, image_latents_mid, latent_padding2, image_latents_last], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.animatediff.pipeline_animatediff_video2video.AnimateDiffVideoToVideoPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        image,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.fuse_qkv_projections
    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.unfuse_qkv_projections
    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        pos_frames:int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        #pdb.set_trace()
        if num_frames==49:
            pos_emb_len = 13
        elif num_frames==97:
            pos_emb_len = 25
        #pdb.set_trace()
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=pos_frames,
            pos_emb_len = pos_emb_len, #13 0r 25
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # def __call__(
    def generate(
        self,
        img_list: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        pos_frames: int =25,
        seed: int=1111,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if num_frames > 49:
            print("The number of frames must be less than 49 for now due to static positional embeddings.")
        self._set_seed(seed)
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=img_list,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        device = self._device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # print(66666666,prompt_embeds.dtype)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        # image = self.video_processor.preprocess(image, height=height, width=width).to(
        #     device, dtype=prompt_embeds.dtype
        # )

        latent_channels = self.transformer.config.in_channels // 2
        # latents, image_latents = self.prepare_latents(
        #     image,
        #     batch_size * num_videos_per_prompt,
        #     latent_channels,
        #     num_frames,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )

        # 初始化潜在空间。总体上就是初始化一个符合正态分布的噪声，
        # num_frames = 49/97
        # 对帧率进行4倍下采样，12+1,24+1, 对宽高进行8倍下采样 480/8, 720/8
        # 噪声s尺寸为[b,frame+1,latent channel,h,w]=[1,13,16,60,90] [1,25,16,60,90]
        if len(img_list)==1:
            img = self.video_processor.preprocess(img_list[0], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            latents, image_latents = self.prepare_latents(
                img,
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        if len(img_list)==2:
            img = self.video_processor.preprocess(img_list[0], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            img_last = self.video_processor.preprocess(img_list[1], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            latents, image_latents = self.prepare_latents_2img(
                img,
                img_last,
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        if len(img_list)==3:
            img = self.video_processor.preprocess(img_list[0], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            img_mid = self.video_processor.preprocess(img_list[1], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            img_last = self.video_processor.preprocess(img_list[2], height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
            latents, image_latents = self.prepare_latents_3img(
                img,
                img_mid,
                img_last,
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        #pdb.set_trace() 
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, num_frames,pos_frames, device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        self.transformer = self.transformer.to(device)
        attn_count=-1
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                attn_count += 1
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                if attn_count >= 18 and attn_count %5 != 0:
                    latent_model_input_todo=latent_model_input[:1]
                    prompt_embeds_todo=prompt_embeds[1:]
                else:
                    latent_model_input_todo=latent_model_input
                    prompt_embeds_todo=prompt_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input_todo.shape[0])
                # predict noise model_output
                ########exp
                # noise_pred = self.transformer(
                #     hidden_states=latent_model_input_todo,
                #     encoder_hidden_states=prompt_embeds_todo,
                #     timestep=timestep,
                #     image_rotary_emb=image_rotary_emb,
                #     return_dict=False,attn_count=attn_count
                # )[0]
                # if attn_count >= 18 and attn_count %5 != 0:
                #     single_output=noise_pred
                #     (bb, tt, cc, hh, ww) = single_output.shape
                #     cond = rearrange(single_output, "B T C H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                #     lf_c, hf_c = fft(cond.float())
                #
                #     if attn_count <= 40:
                #         self.delta_lf = self.delta_lf * 1.1
                #     if attn_count >= 30:
                #         self.delta_hf = self.delta_hf * 1.1
                #
                #     new_hf_uc = self.delta_hf + hf_c
                #     new_lf_uc = self.delta_lf + lf_c
                #
                #     combine_uc = new_lf_uc + new_hf_uc
                #     combined_fft = torch.fft.ifftshift(combine_uc)
                #     recovered_uncond = torch.fft.ifft2(combined_fft).real
                #     recovered_uncond = rearrange(recovered_uncond.to(single_output.dtype), "(B T) C H W -> B T C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                #     output = torch.cat([single_output, recovered_uncond])
                #
                #     noise_pred=output
                # elif attn_count>=16:
                #     output=noise_pred
                #     (bb, tt, cc, hh, ww) = output.shape
                #     cond = rearrange(output[0:1].float(), "B T C H W -> (B T) C H W", B=bb // 2, C=cc, T=tt, H=hh, W=ww)
                #     uncond = rearrange(output[1:2].float(), "B T C H W -> (B T) C H W", B=bb // 2, C=cc, T=tt, H=hh, W=ww)
                #
                #     lf_c, hf_c = fft(cond)
                #     lf_uc, hf_uc = fft(uncond)
                #
                #     self.delta_lf = lf_uc - lf_c
                #     self.delta_hf = hf_uc - hf_c

                #######base
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]


                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        self.transformer=self.transformer.cpu()
        torch.cuda.empty_cache()

        if not output_type == "latent":
            self.vae=self.vae.to(device)
            video = self.decode_latents(latents)
            self.vae=self.vae.cpu()
            torch.cuda.empty_cache()
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
