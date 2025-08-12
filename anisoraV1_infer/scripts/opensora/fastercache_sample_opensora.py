import argparse
import os
import time

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from omegaconf import OmegaConf
from tqdm import tqdm

from fastercache.models.opensora import RFLOW, OpenSoraVAE_V1_2, STDiT3_XL_2, T5Encoder, text_preprocessing
from fastercache.models.opensora.datasets import get_image_size, get_num_frames, save_sample
from fastercache.models.opensora.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from fastercache.utils.utils import all_exists, create_logger, merge_args, set_seed, str_to_dtype

from fastercache.models.opensora.utils import auto_grad_checkpoint, load_checkpoint

from fastercache.models.opensora.modules import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from einops import rearrange

@torch.no_grad()
def fastercache_STDiT3Block_forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        timestep=None,
        block_id=None,
        counter=None,
    ):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        if counter>7 and counter%2==0:
            x_m = self.attn_cache[1][:x.shape[0]] +(self.attn_cache[1][:x.shape[0]] - self.attn_cache[0][:x.shape[0]]) * (counter - 7)/20.0 * 0.9 
        else:
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
            
            if counter == 5:
                self.attn_cache = [x_m, x_m]
            elif counter > 5:
                self.attn_cache[0] = self.attn_cache[1]
                self.attn_cache[1] = x_m
                
        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        x_cross = self.cross_attn(x, y, mask)
        x = x + x_cross

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)
        return x


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

@torch.no_grad()
def fastercache_model_forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
    dtype = self.x_embedder.proj.weight.dtype
    tms = timestep.to(dtype)

    if tms[0] == 1000:
        self.counter = 0
    self.counter += 1

    if self.counter % 5 !=0 and self.counter>11:
        x_single, timestep_single, y_single, mask_single, x_mask_single = x[:1], timestep[:1], y[:1], mask, x_mask[:1]
        single_output = self.fastercache_model_forward_single(x_single, timestep_single, y_single, mask_single, x_mask_single, fps, height, width, self.counter, **kwargs)

        (bb, cc, tt, hh, ww) = single_output.shape
        cond = rearrange(single_output, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
        lf_c, hf_c = fft(cond)
        
        if self.counter >= 20:
            self.cache_uncond_delta = self.cache_uncond_delta * 1.15
        if self.counter <= 25:
            self.cache_uncond_delta_low = self.cache_uncond_delta_low * 1.15

        new_hf_uc = self.cache_uncond_delta + hf_c
        new_lf_uc = self.cache_uncond_delta_low + lf_c

        combine_uc = new_lf_uc + new_hf_uc
        combined_fft = torch.fft.ifftshift(combine_uc)
        recovered_uncond = torch.fft.ifft2(combined_fft).real
        recovered_uncond = rearrange(recovered_uncond, "(B T) C H W -> B C T H W", B=bb, C=cc, T=tt, H=hh, W=ww)
        output = torch.cat([single_output,recovered_uncond])
        return output
    else:
        output = self.fastercache_model_forward_single(x, timestep, y, mask, x_mask, fps, height, width, self.counter, **kwargs)

        if self.counter>=10:
            (bb, cc, tt, hh, ww) = output.shape
            cond = rearrange(output[0:1], "B C T H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)
            uncond = rearrange(output[1:2], "B C T H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)

            lf_c, hf_c = fft(cond)
            lf_uc, hf_uc = fft(uncond)

            self.cache_uncond_delta = hf_uc - hf_c
            self.cache_uncond_delta_low = lf_uc - lf_c
        return output


@torch.no_grad()
def fastercache_model_forward_single(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, counter=None, **kwargs):
    dtype = self.x_embedder.proj.weight.dtype
    B = x.size(0)
    x = x.to(dtype)
    timestep = timestep.to(dtype)
    y = y.to(dtype)

    # === get pos embed ===
    _, _, Tx, Hx, Wx = x.size()
    T, H, W = self.get_dynamic_size(x)
    S = H * W
    base_size = round(S**0.5)
    resolution_sq = (height[0].item() * width[0].item()) ** 0.5
    scale = resolution_sq / self.input_sq_size
    pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

    # === get timestep embed ===
    t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
    fps = self.fps_embedder(fps.unsqueeze(1), B)
    t = t + fps
    t_mlp = self.t_block(t)
    t0 = t0_mlp = None
    if x_mask is not None:
        t0_timestep = torch.zeros_like(timestep)
        t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
        t0 = t0 + fps
        t0_mlp = self.t_block(t0)

    # === get y embed ===
    if self.config.skip_y_embedder:
        y_lens = mask
        if isinstance(y_lens, torch.Tensor):
            y_lens = y_lens.long().tolist()
    else:
        y, y_lens = self.encode_text(y, mask)

    # === get x embed ===
    x = self.x_embedder(x)  # [B, N, C]
    x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
    x = x + pos_emb

    x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

    # === blocks ===
    block_id = 0
    for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
        x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep, block_id, counter)
        block_id+=1
        x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, timestep, block_id, counter)
        block_id+=1

    # === final layer ===
    x = self.final_layer(x, t, x_mask, t0, T, S)
    x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

    # cast to float32 for better accuracy
    x = x.to(torch.float32)
    return x



@torch.no_grad()
def main(args):
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == dtype ==
    dtype = str_to_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    device = f"cuda:{torch.cuda.current_device()}"
    set_seed(seed=args.seed)

    # == init logger ==
    logger = create_logger()
    logger.info(f"Inference configuration: {args}\n")
    verbose = args.verbose
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = T5Encoder(
        from_pretrained="DeepFloyd/t5-v1_1-xxl", model_max_length=300, device=device, shardformer=args.enable_t5_speedup
    )
    vae = (
        OpenSoraVAE_V1_2(
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
            micro_frame_size=17,
            micro_batch_size=4,
        )
        .to(device, dtype)
        .eval()
    )

    # == prepare video size ==
    image_size = args.image_size
    if image_size is None:
        resolution = args.resolution
        aspect_ratio = args.aspect_ratio
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(args.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        STDiT3_XL_2(
            from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=True,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
        )
        .to(device, dtype)
        .eval()
    )

    for _name, _module in model.named_modules():
        if _module.__class__.__name__=='STDiT3':
            _module.__class__.forward  = fastercache_model_forward
            _module.__class__.fastercache_model_forward_single = fastercache_model_forward_single
        if _module.__class__.__name__=='STDiT3Block':
            _module.__class__.__call__  = fastercache_STDiT3Block_forward

    text_encoder.y_embedder = model.y_embedder 

    # == build scheduler ==
    scheduler = RFLOW(use_timestep_transform=True, num_sampling_steps=30, cfg_scale=7.0)

    def load_prompts(prompt_path, start_idx=None, end_idx=None):
        with open(prompt_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        prompts = prompts[start_idx:end_idx]
        return prompts
    prompts = None
    if prompts is None:
        assert args.prompt_path is not None
        prompts = load_prompts(args.prompt_path)

    # ======================================================
    # inference
    # ======================================================

    # == prepare reference ==
    reference_path = args.reference_path if args.reference_path is not None else [""] * len(prompts)
    mask_strategy = args.mask_strategy if args.mask_strategy is not None else [""] * len(prompts)
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = args.fps
    save_fps = fps // args.frame_interval
    multi_resolution = args.multi_resolution
    batch_size = args.batch_size
    num_sample = args.num_sample
    loop = args.loop
    condition_frame_length = args.condition_frame_length
    condition_frame_edit = args.condition_frame_edit
    align = args.align

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    prompt_as_path = args.prompt_as_path

    # == Iter over all samples ==
    z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)

    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_idx=idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=args.aes,
                    flow=args.flow,
                    camera_motion=args.camera_motion,
                )

            # clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )

                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            # == save samples ==
            if coordinator.is_master():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):
                        video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path+'_'+str(i),
                        verbose=verbose >= 2,
                    )
                    if save_path.endswith(".mp4") and args.watermark:
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)

    logger.info("Inference finished.")
    logger.info("Saved samples to %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--config", default=None, type=str, help="path to config yaml")
    parser.add_argument("--seed", default=942, type=int, help="seed for reproducibility")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--flash-attn", action="store_true", help="enable flash attention")
    parser.add_argument("--enable_t5_speedup", action="store_true", help="enable t5 speedup")
    parser.add_argument("--resolution", default=None, type=str, help="resolution")
    parser.add_argument("--multi-resolution", default=None, type=str, help="multi resolution")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")

    # output
    parser.add_argument("--save-dir", default="./samples/opensora", type=str, help="path to save generated samples")
    parser.add_argument("--num-sample", default=1, type=int, help="number of samples to generate for one prompt")
    parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
    parser.add_argument("--verbose", default=2, type=int, help="verbose level")

    # prompt
    parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
    parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
    parser.add_argument("--llm-refine", action="store_true", help="enable LLM refine")

    # image/video
    parser.add_argument("--num-frames", default=None, type=str, help="number of frames")
    parser.add_argument("--fps", default=24, type=int, help="fps")
    parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
    parser.add_argument("--frame-interval", default=1, type=int, help="frame interval")
    parser.add_argument("--aspect-ratio", default=None, type=str, help="aspect ratio (h:w)")
    parser.add_argument("--watermark", action="store_true", help="watermark video")

    # hyperparameters
    parser.add_argument("--num-sampling-steps", default=30, type=int, help="sampling steps")
    parser.add_argument("--cfg-scale", default=7.0, type=float, help="balance between cond & uncond")

    # reference
    parser.add_argument("--loop", default=1, type=int, help="loop")
    parser.add_argument("--align", default=None, type=int, help="align")
    parser.add_argument("--condition-frame-length", default=5, type=int, help="condition frame length")
    parser.add_argument("--condition-frame-edit", default=0.0, type=float, help="condition frame edit")
    parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
    parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    parser.add_argument("--aes", default=None, type=float, help="aesthetic score")
    parser.add_argument("--flow", default=None, type=float, help="flow score")
    parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")

    args = parser.parse_args()

    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)

