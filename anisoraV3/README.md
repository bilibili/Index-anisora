##  🚀 Quick Started

## Video Demos

<div align="center">
    <video src="https://github.com/user-attachments/assets/a207d43e-26f6-445b-883e-9b28c129607f" controls width="60%" poster=""></video>
</div>

<div align="center">
    <video src="https://github.com/user-attachments/assets/4351fc5e-f7fd-456b-807e-82fdcb321de2" controls width="60%" poster=""></video>
</div>


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

Please download AnisoraV3 checkpoints from [Huggingface](https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3)

```bash
git lfs install
git clone https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3
```


### 3. Inference

#### Single-GPU Inference 

```bash
python generate-pi-i2v-any.py --task i2v-14B --size 1280*720  --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos_any --prompt data/inference_any.txt --base_seed 4096 --frame_num 81
```

#### Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 --master_port 43210 generate-pi-i2v-any.py --task i2v-14B --size 1280*720  --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos_any --prompt data/inference_any.txt --dit_fsdp --t5_fsdp --ulysses_size 2 --base_seed 4096 --frame_num 81 --sample_steps 8 --sample_shift 5 --sample_guide_scale 1
```

### 4. Inference

### 360-Degree Character Rotation
#### Single-GPU Inference 

```bash
python generate-pi-i2v-any.py --task i2v-14B --size 1280*720 --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos_360 --prompt data/inference_360.txt --base_seed 4096 --frame_num 81
```

#### Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 --master_port 43210 generate-pi-i2v-any.py --task i2v-14B --size 1280*720 --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos_360 --prompt data/inference_360.txt --dit_fsdp --t5_fsdp --ulysses_size 2 --base_seed 4096 --frame_num 81 --sample_steps 8 --sample_shift 5 --sample_guide_scale 1
```

Where,

    The --prompt file contains prompts and image paths and image position in time dimension（0: first frame，0.5: mid frame，1: last frame） for all cases (format: one line per case, image_path@@image_prompt&&image_position), see `data/inference_any.txt` :
        An aesthetic score suffix can be fixed to 5.5, motion score is recommended between 1 and 2 (higher means more motion).
        
    --image specifies the output folder  
    
    --nproc_per_node and --ulysses_size should both be set to the number of GPUs used for multi-GPU inference.  
    
    --ckpt_dir is the root directory of model checkpoint.  
    
    --frame_num is the number of frames to infer, at 16fps, 49 frames equals about 3 seconds, must satisfy F=8x+1 (x∈Z), 360-degree character rotation recommended 5s.  