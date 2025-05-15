import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf
from copy import deepcopy
import torch.nn.functional as F

from sat.helpers import print_rank0
import torch
from torch import nn

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
import gc
from sat import mpu
import random
import numpy as np

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        # skip vae encode config
        self.is_training_from_vae_encoded_data = args.training_from_vae_encoded_data
        self.training_from_vae_encoded_condition = False
        self.using_mock_vae_encoded_data = args.mock_vae_encoded_data
        # if self.using_mock_vae_encoded_data:
        #     assert hasattr(args, "mock_vae_encoded_config"), "Using mock vae data need using mock-vae-encoded-config."
        #     self.vae_chan_out_dim = args.mock_vae_encoded_config["z_channels"] if args.mock_vae_encoded_config["double_z"]  \
        #                 else args.mock_vae_encoded_config["z_channels"] // 2

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))

        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def disable_untrainable_params(self):
        total_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            flag = False
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all":
                    flag = True
                    break

            lora_prefix = ["matrix_A", "matrix_B"]
            for prefix in lora_prefix:
                if prefix in n:
                    flag = False
                    break

            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    # def add_noise_to_first_frame(self, image):
    #     sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
    #     sigma = torch.exp(sigma).to(image.dtype)
    #     image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    #     image = image + image_noise
    #     return image

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch) #torch.Size([1, 33, 3, 720, 1088]) torch.Size([1, 16, 11, 90, 136])
        x_video = batch['mp4_real'].to(self.dtype)
        if self.noised_image_input:
            if not self.training_from_vae_encoded_condition:
                if np.random.randint(2)>0:
                    index_ = np.random.randint(x_video.shape[1])
                else:
                    index_ = 0
                if np.random.randint(2)>0:
                    index_2 = np.random.randint(x_video.shape[1])
                else:
                    index_2 = None
                    
                bianjie_T = x_video.shape[1]-4
                if index_==0 or index_ >= bianjie_T:
                    index_tmp = (index_-1)//4+1 
                    image=self.scale_factor*x[:, :, index_tmp:index_tmp+1]
                else:
                    x_video_for1 = x_video.permute(0, 2, 1, 3, 4).contiguous()#torch.Size([1, 3, 33, 720, 1088])
                    image = self.encode_first_stage(x_video_for1[:, :, index_:index_+1], batch)
                    
                if index_==0:
                    index_tmp = (index_-1)//4+1
                    image_offline =self.scale_factor*x[:, :, index_tmp:index_tmp+1]
                    image_online = self.encode_first_stage( x_video.permute(0, 2, 1, 3, 4).contiguous()[:, :, index_:index_+1], batch)
                    image_offline_all = torch.cat([image_offline, image_online], 0)

                    
                if index_2 is not None:
                    if index_2==0 or index_2 >= bianjie_T:
                        index_2_tmp = (index_2-1)//4+1 
                        image_2=self.scale_factor*x[:, :, index_2_tmp:index_2_tmp+1]
                    else:
                        x_video_for2 = x_video.permute(0, 2, 1, 3, 4).contiguous()#torch.Size([1, 3, 33, 720, 1088])
                        image_2 = self.encode_first_stage(x_video_for2[:, :, index_2:index_2+1], batch)
            else:
                if np.random.randint(2)>0:
                    index_ = np.random.randint(x.shape[2])
                else:
                    index_ = 0
                if np.random.randint(2)>0:
                    index_2 = np.random.randint(x.shape[2])
                else:
                    index_2 = None
                image=self.scale_factor*x[:, :, index_:index_+1]
                if index_2 is not None:
                    image_2=self.scale_factor*x[:, :, index_2:index_2+1]
        if not self.using_mock_vae_encoded_data:
            if self.lr_scale is not None:
                # 用于调整视频帧的分辨率以适应网络的输入要求
                lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
                lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
                if not self.is_training_from_vae_encoded_data:
                    lr_z = self.encode_first_stage(lr_x, batch)
                batch["lr_input"] = lr_z
            if not self.is_training_from_vae_encoded_data:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                x = self.encode_first_stage(x, batch)
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            else:
                x*=self.scale_factor
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.noised_image_input:
            image = image.permute(0, 2, 1, 3, 4).contiguous()#
            if index_2 is not None:
                image_2 = image_2.permute(0, 2, 1, 3, 4).contiguous()#
            if self.noised_image_all_concat:
                image = image.repeat(1, x.shape[1], 1, 1, 1) #TODO: 改回去
            else:
                if not self.training_from_vae_encoded_condition:
                    index_ = (index_-1)//4+1
                    if index_2 is not None:
                        index_2 = (index_2-1)//4+1
                fe_0 = torch.zeros_like(x[:, 0:index_])
                fe_1 = torch.zeros_like(x[:, index_+1:])
                if index_2 is not None:
                    if index_2>index_:
                        fe_1_A = torch.zeros_like(x[:, index_+1:index_2])
                        fe_1_B = torch.zeros_like(x[:, index_2+1:])
                        image = torch.concat([fe_0, image, fe_1_A, image_2, fe_1_B], dim=1)
                    elif index_2<index_:
                        fe_0_A = torch.zeros_like(x[:, 0:index_2])
                        fe_0_B = torch.zeros_like(x[:, index_2+1:index_])
                        image = torch.concat([fe_0_A, image_2, fe_0_B, image, fe_1], dim=1)
                    elif index_2==index_:
                        image = torch.concat([fe_0, image, fe_1], dim=1)
                else:
                    try:
                        image = torch.concat([fe_0, image, fe_1], dim=1)
                    except:
                        print('fe_0', fe_0.shape, 'image', image.shape, 'fe_1', fe_1.shape)
                        image = torch.concat([fe_0, image, fe_1], dim=1)
            if random.random() < self.noised_image_dropout:
                image = torch.zeros_like(image)
            batch["concat_images"] = image
        gc.collect()
        # torch.cuda.empty_cache()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                use_cp = False
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return x * self.scale_factor  # already encoded

        use_cp = False

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples].contiguous())
                # using no context parallel for debug
                # out = self.first_stage_model.encode_v0(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        prefix=None,
        concat_images=None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None
        scale_emb = None

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb)
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        only_log_video_latents=False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
        samples = samples.permute(0, 2, 1, 3, 4).contiguous()
        if only_log_video_latents:
            latents = 1.0 / self.scale_factor * samples
            log["latents"] = latents
        else:
            samples = self.decode_first_stage(samples).to(torch.float32)
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            log["samples"] = samples
        return log
