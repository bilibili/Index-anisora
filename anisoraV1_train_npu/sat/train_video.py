import os
import argparse
from functools import partial
from PIL import Image
import numpy as np
import torch.distributed
from omegaconf import OmegaConf
import imageio

from sat.helpers import print_rank0
import torch
if os.getenv("ACCELERATOR") == "npu":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
    print_rank0("import torch_npu\n")
else:
    print_rank0("Warning: Missing torch_npu\n")

from sat import mpu
from sat.training.deepspeed_training import training_main

from sgm.util import get_obj_from_str, isheatmap, exists
from transformer_sp.util import (get_sequence_parallel_group,get_sequence_parallel_size,get_sequence_parallel_group_rank)

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args, process_config_to_args

from einops import rearrange, repeat

try:
    import wandb
except ImportError:
    print("warning: wandb not installed")


def print_debug(args, s):
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)


def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)
        if args is not None and args.wandb:
            wandb.log(
                {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration + 1
            )


def log_video(batch, model, args, only_log_video_latents=False):
    texts = batch["txt"]
    text_save_dir = os.path.join(args.save, "video_texts")
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    if torch.distributed.get_rank() == 0:
        root = os.path.join(args.save, "video")

        if only_log_video_latents:
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos["latents"], os.path.join(path, "latent.pt"))
        else:
            for k in videos:
                N = videos[k].shape[0]
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            num_frames = batch["num_frames"][0]
            fps = batch["fps"][0].cpu().item()
            if only_log_video_latents:
                root = os.path.join(root, "latents")
                filename = "{}_gs-{:06}".format("latents", args.iteration)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                os.makedirs(path, exist_ok=True)
                torch.save(videos["latents"], os.path.join(path, "latents.pt"))
            else:
                for k in videos:
                    samples = (videos[k] + 1.0) / 2.0
                    filename = "{}_gs-{:06}".format(k, args.iteration)

                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)



def broad_cast_batch(batch):
    # mp_size = mpu.get_model_parallel_world_size()# how many group
    # global_rank = torch.distributed.get_rank() // mp_size# group number
    # src = global_rank * mp_size# the first of group

    sp_size = get_sequence_parallel_size()# how many group
    global_src_rank = get_sequence_parallel_group_rank() * sp_size# group number * how many group = the first of group
    group=get_sequence_parallel_group()

    if batch["mp4"] is not None:
        broadcast_shape = [batch["mp4"].shape, batch["fps"].shape, batch["num_frames"].shape, batch["mp4_real"].shape]
    else:
        broadcast_shape = None

    txt = [batch["txt"], broadcast_shape]
    rank=torch.distributed.get_rank()
    # if rank % sp_size != 0:
    # torch.distributed.broadcast_object_list(txt, src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast_object_list(txt, src=global_src_rank, group=group)
    batch["txt"] = txt[0]

    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]
    mp4_real_shape = txt[1][3]

    # if mpu.get_model_parallel_rank() != 0:
    if rank%sp_size != 0:# if the first of group,because broadcast need the same shape
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda",dtype=torch.float16)
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(num_frames_shape, device="cuda", dtype=torch.long)
        batch["mp4_real"] = torch.zeros(mp4_real_shape, device="cuda",dtype=torch.float32)

    # torch.distributed.broadcast(batch["mp4"], src=src, group=mpu.get_model_parallel_group())
    # torch.distributed.broadcast(batch["fps"], src=src, group=mpu.get_model_parallel_group())
    # torch.distributed.broadcast(batch["num_frames"], src=src, group=mpu.get_model_parallel_group())

    torch.distributed.broadcast(batch["mp4"], src=global_src_rank, group=group)
    torch.distributed.broadcast(batch["fps"], src=global_src_rank, group=group)
    torch.distributed.broadcast(batch["num_frames"], src=global_src_rank, group=group)
    torch.distributed.broadcast(batch["mp4_real"], src=global_src_rank, group=group)

    return batch

def forward_step_eval(data_iterator, model, args, timers, only_log_video_latents=False, data_class=None):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch_video = next(data_iterator)
        timers("data loader").stop()

        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]
            batch_video["mp4"] = batch_video["mp4"].view(-1, *batch_video["mp4"].shape[2:])
            txt = []
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt

        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None}
    broad_cast_batch(batch_video)
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)

    batch_video["global_step"] = args.iteration
    loss, loss_dict = model.shared_step(batch_video)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict


def forward_step(data_iterator, model, args, timers, data_class=None):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch = next(data_iterator)
        timers("data loader").stop()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        if torch.distributed.get_rank() == 0:
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                config = OmegaConf.merge(*configs)
                os.makedirs(args.save, exist_ok=True)
                OmegaConf.save(config=config, f=os.path.join(args.save, "training_config.yaml"))
    else:
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None}

    batch["global_step"] = args.iteration

    broad_cast_batch(batch)

    loss, loss_dict = model.shared_step(batch)

    return loss, loss_dict


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    if args.mock_vae_encoded_data:
        os.environ['DATASET_USING_MOCK_DATA'] = "true"
        assert hasattr(args, "mock_vae_encoded_config"), "Using mock vae data need using mock-vae-encoded-config."
        os.environ['MOCK_VAE_ENCODERED_Z_CHANNELS'] = str(args.mock_vae_encoded_config["z_channels"])
        os.environ['MOCK_VAE_ENCODERED_DOUBLE_Z'] = str(args.mock_vae_encoded_config["double_z"])


    args = argparse.Namespace(**vars(args), **vars(known))
    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    # import yaml
    # configs = []
    # for config in args.base:
    #     with open(config, "r") as f:
    #         base_config = yaml.safe_load(f)
    #     configs.append(base_config)
    # args.log_config = configs

    for k in args.__dict__:
        print_rank0(k + ": " + str(args.__dict__[k]), log_file="param.txt")

    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class),
        forward_step_eval=partial(
            forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents
        ),
        create_dataset_function=create_dataset_function,
    )
