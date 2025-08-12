import os
os.environ["PATH"] += os.pathsep + "/usr/bin"
import json
import time

import click
import numpy as np
import torch

from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)
from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video

pipeline = None
model_dir_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False


def configure_model(model_dir_path_, cpu_offload_):
    global model_dir_path, cpu_offload
    model_dir_path = model_dir_path_
    cpu_offload = cpu_offload_


def load_model():
    global num_gpus, pipeline, model_dir_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(model_path=f"{MOCHI_DIR}/dit.safetensors", model_dtype="bf16"),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/vae.safetensors",
                model_stats_path=f"{MOCHI_DIR}/vae_stats.json",
            ),
        )
        if num_gpus > 1:
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
            kwargs["tiled_decode"] = True
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
):
    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025, linear_steps=5)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    with progress_bar(type="tqdm"):
        args['acc'] = True
        final_frames = pipeline(**args)

        final_frames = final_frames[0]

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32

        os.makedirs("outputs_mochi/", exist_ok=True)
        output_path = os.path.join("outputs_mochi/", prompt[:50].replace(' ','_').replace('\n','')+'_'+f"output_{int(time.time())}.mp4")


        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        json.dump(args, open(json_path, "w"), indent=4)

        return output_path

from textwrap import dedent


original_prompt=[
"""A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
filled with lemons and sprigs of mint against a peach-colored background. 
The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
Additional lemons, one halved, are scattered around the base of the bowl. 
The even lighting enhances the vibrant colors and creates a fresh, 
inviting atmosphere.""",
"""a serene winter scene in a forest. The forest is blanketed in a thick layer of snow, which has settled on the branches of the trees, creating a canopy of white. The trees, a mix of evergreens and deciduous, stand tall and silent, their forms partially obscured by the snow. The ground is a uniform white, with no visible tracks or signs of human activity. The sun is low in the sky, casting a warm glow that contrasts with the cool tones of the snow. The light filters through the trees, creating a soft, diffused illumination that highlights the texture of the snow and the contours of the trees. The overall style of the scene is naturalistic, with a focus on the tranquility and beauty of the winter landscape.""",
"""a dynamic interaction between the ocean and a large rock. The rock, with its rough texture and jagged edges, is partially submerged in the water, suggesting it is a natural feature of the coastline. The water around the rock is in motion, with white foam and waves crashing against the rock, indicating the force of the ocean's movement. The background is a vast expanse of the ocean, with small ripples and waves, suggesting a moderate sea state. The overall style of the scene is a realistic depiction of a natural landscape, with a focus on the interplay between the rock and the water.""",
"""3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest.""",
"""A fancy modern house on the ocean is on fire.""",
"""A serene waterfall cascading down moss-covered rocks, its soothing sound creating a harmonious symphony with nature.""",
"""Slow pan upward of blazing oak fire in an indoor fireplace.""",
"""Sunset over the sea.""",
"""A fast-tracking shot down an suburban residential street lined with trees. Daytime with a clear blue sky. Saturated colors, high contrast.""",
"""An aerial shot of a lighthouse stands tall on a rocky cliff, its beacon cutting through the early dawn, waves crash against the rocks below.""",
"""The majestic beauty of a waterfall cascading down a cliff into a serene lake.""",
"""Timelapse of the northern lights dancing across the Arctic sky, stars twinkling, snow-covered landscape.""",
]

DEFAULT_PROMPT = ""
@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=12345, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, cpu_offload
):
    configure_model(model_dir, cpu_offload)
    for prompt in original_prompt:
        prompt = dedent(prompt)
        output = generate_video(
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            seed,
            cfg_scale,
            num_steps,
        )
        click.echo(f"Video generated at: {output}")


if __name__ == "__main__":
    generate_cli()

