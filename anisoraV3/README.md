##  🚀 Quick Started

### 1. Environment Set Up

```bash
cd anisoraV3
conda create -n wan_gpu python=3.10
conda activate wan_gpu
pip install -r req-fastvideo.txt
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Weights

Please download AnisoraV3 checkpoints from [Huggingface](https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3.1)

```bash
git lfs install
git clone https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3.1
```


### 3. Inference

#### Single-GPU Inference 

```bash
python generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280*720  \
    --ckpt_dir Wan2.1-I2V-14B-480P \
    --image output_videos_any \
    --prompt data/inference_any.txt \
    --base_seed 4096 \
    --frame_num 81
```

#### Multi-GPU Inference

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port 43210 \
    generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280*720  \
    --ckpt_dir Wan2.1-I2V-14B-480P \
    --image output_videos_any \
    --prompt data/inference_any.txt \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --base_seed 4096 \
    --frame_num 81 \
    --sample_steps 8 \
    --sample_shift 5 \
    --sample_guide_scale 1
```

### 4. Inference

### 360-Degree Character Rotation
#### Single-GPU Inference 

```bash
python generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280x720 \
    --ckpt_dir Wan2.1-I2V-14B-480P \
    --image output_videos_360 \
    --prompt data/inference_360.txt \
    --base_seed 4096 \
    --frame_num 81
```

#### Multi-GPU Inference

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port 43210 \
    generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir Wan2.1-I2V-14B-480P \
    --image output_videos_360 \
    --prompt data/inference_360.txt \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --base_seed 4096 \
    --frame_num 81 \
    --sample_steps 8 \
    --sample_shift 5 \
    --sample_guide_scale 1
```

Where,

    --prompt 
    
    The prompt File Format: image_path@@prompt&&image_position

        One line per case
        image_position: Temporal position (0=first frame, 0.5=mid frame, 1=last frame)
        Example (from data/inference_any.txt)
    
    --image specifies the output folder  
    
    --nproc_per_node and --ulysses_size should both be set to the number of GPUs used for multi-GPU inference.  
    
    --ckpt_dir is the root directory of model checkpoint.  
    
    --frame_num is the number of frames to infer, at 16fps.
        81 frames equals about 5 seconds, must satisfy F=8x+1 (x∈Z)
        360-degree character rotation recommended 5s, to ensure a full circle of 360 degrees.  

    
#### Prompt Format Specification

Basic Structure

```bash
[Video description in English Better] + aesthetic score: X.X. motion score: X.X. There is no text in the video.
```

| Parameter | Recommended Range | Description |
|-------|-------|-------|
| Aesthetic Score | 5.0-7.0 | Controls visual quality and cinematic appeal (higher = more cinematic) |
| Motion Score | 2.0-4.0 | Controls movement intensity (higher values = more dynamic motion) |
| No Text Clause | Mandatory | Prevents unwanted captions or text overlays in generated videos |

example:  A drone chase sequence through neon-lit city streets at night. aesthetic score: 5.5. motion score: 4.0. There is no text in the video.

