##  🚀 Quick Started

### 1. Environment Setup

```bash
cd anisoraV1_infer 
conda create -n ani_infer python=3.10
conda activate ani_infer
pip install -r requirements.txt
```

or you can follow https://github.com/Vchitect/FasterCache to setup the environment.

### 2. Download Pretrained Weights

Please download the text_encoder and VAE from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/CogVideoX_VAE_T5) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put them in `./pretrained_models/`. 

Please download 5B model weights from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/5B) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put it in `./ckpt/`.

### 3. Inference

For, A100 , you can set `offload=0`:
```bash
offload=0 python scripts/cogvideox/fastercache_sample_cogvideox_sp.py --base configs/cogvideox/fastercache_sample_5b.yaml
```

you can also use multiple GPU to inference
```bash
offload=0 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=25000 scripts/cogvideox/fastercache_sample_cogvideox_sp.py --base configs/cogvideox/fastercache_sample_5b.yaml
```

For, 4x4090, you must set `offload=1`:
```bash
offload=1 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=25000 scripts/cogvideox/fastercache_sample_cogvideox_sp.py --base configs/cogvideox/fastercache_sample_5b4090.yaml
```

### 4. Config Description

You can modify the yaml file in the command line to use other pretrained models or generate height/width/frames video.

input_file: the prompt file. the format must be: prompt@@image_path. You can adjust the "motion score" in the tail to control the movement intensity, typically within the range of 0.5 to 1.5. The higher the score, the stronger the movement. (see prompt_demo.txt)

args/load: the pretrained diffusion model path

model/first_stage_config/params/ckpt_path: the CogvideoX vae model path

model/conditioner_config/params/params/model_dir: the T5 text encoder model dir

model/network_config/params/latent_width(latent_height): If they are 160 * 90, a 1280 * 720 video will be generated.

model/network_config/params/num_frames: If it's 49, a 49 frames video (49//16=3s) will be generated, while args/sampling_num_frames must be 13 (13=49//4+1).




## 📁 System Notes

- **Supports up to SP4**
- **2 × RTX 4090** runs **OOM** during decoding
- **RTX 4090** only supports up to **640×1088** resolution  
  - Therefore, for testing:  
    - 4090 uses **640×1088** 
    - A800 uses standard **720×1280**

---

## 🍎 Performance Results

| Configuration | Time (seconds) |
|---------------|----------------|
| 4×RTX 4090     | 341            |
| 4×A800         | 224            |
| 2×A800         | 356            |
| 1×A800         | 609            |


