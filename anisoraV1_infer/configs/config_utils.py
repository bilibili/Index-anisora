import argparse
import json
import os
from glob import glob

def parse_args(training=False):
    parser = argparse.ArgumentParser()
    # model config
    # parser.add_argument("--config", default='./configs/infer.yaml', type=str, help="model config file path")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    #personal/Work/cogvideo_i2v_server/cogvideox_4090/cogvideox/diffusers_ckpts/CogvideoX_169
    parser.add_argument(
        "--model_path", type=str, default="/DATA/workshop/personal/Work/cogvideo_i2v_server/cogvideox_4090/cogvideox/tools/sat_ckpts/CogvideoX_11/", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16', 'bfloat16')"
    )
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="bf16",
        choices=["int8", "fp8", "bf16"],
        help="The quantization scheme to use (int8, fp8)",
    )
    parser.add_argument(
        "--generate_type", type=str, default="i2v", choices=["t2v", "i2v", "v2v"], help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    return parser.parse_args()


# def read_config(config_path):
#     cfg = Config.fromfile(config_path)
#     return cfg



def parse_configs(training=False):
    args = parse_args(training)
    #cfg = read_config(args.config)
    #cfg = merge_args(cfg, args, training)
    #return cfg
    return args
