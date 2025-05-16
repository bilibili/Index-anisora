##  🚀 Quick Started

### 1. Environment Set Up

```bash
cd anisoraV2_gpu
conda create -n wan_gpu python=3.10
conda activate wan_gpu
pip install -r req-fastvideo.txt
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Weights

Please download AnisoraV2 checkpoints from [Huggingface] comming soon....

```bash
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
```

### 2. Data Preprocessing

1. Prepare video folder and prompt file (see `data/mp4root`,` data/train_data_prompts.txt`).
2. Modify input path, resolution, fps, and checkpoint root in the 4 feature extraction scripts under `preprocess/`.
3. Run the scripts to generate `.pt` feature files.
4. Organize training data following the format in `data/data.json`, place features in `data/test_pts/`

### 3. Training

```bash
bash scripts/finetune/finetune_wan.sh
```

### 4. Inference

#### Single-GPU Inference 

```bash
python generate-pi-i2v.py --task i2v-14B --size 960*544 --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos --prompt data/inference.txt --base_seed 4096 --frame_num 49
```

#### Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 --master_port 43210 generate-pi-i2v.py --task i2v-14B --size 960*544 --ckpt_dir Wan2.1-I2V-14B-480P --image output_videos --prompt data/inference.txt --dit_fsdp --t5_fsdp --ulysses_size 2 --base_seed 4096 --frame_num 49
```

Where,
    The --prompt file contains prompts and image paths for all cases (format: one line per case, image_path@@image_prompt), see `data/inference.txt` :
        An aesthetic score suffix can be fixed to 5.5, motion score is recommended between 1 and 2 (higher means more motion), adding "The video has no subtitles." means removing subtitles.
    --image specifies the output folder.
    --nproc_per_node and --ulysses_size should both be set to the number of GPUs used for multi-GPU inference.
    --ckpt_dir is the root directory of model checkpoint.
    --frame_num is the number of frames to infer, at 16fps, 49 frames equals about 3 seconds, must satisfy F=8x+1 (x∈Z).

### 5. OSS Inference Acceleration (≈ 2.5x Speed)

Follow [OptimalSteps](https://github.com/bebebe666/OptimalSteps).
By step search, a subset of teacher steps is selected to make the student video as similar as possible to the teacher's output.  
Tested on the model, it can be stably reduced from 40 steps (baseline) to 16 steps with only a minimal increase in bad cases.

#### Teacher Distillation (96 steps -> 16 steps)

Run 9 samples for 9 times, each run will save one step list and exit after processing the first prompt in txt.
Each sample requires running 96*(16+1) steps — very slow, be prepared.

**Example command**:
```bash
python generate-pi-i2v-myinfer-oss-tea.py --task i2v-14B --size 960*544 --ckpt_dir Wan2.1-I2V-14B-480P --image output/oss_tea_res0 --prompt data/inference.txt --norm 2 --frame_type "4" --channel_type "all" --base_seed 42 --frame_num 49 --sample_steps 96 --student_steps 16
```

Where,
    --sample_steps is the number of teacher steps.
    --student_steps is the number of student steps to distill into.

After collecting 9 step lists, calculate the median:
```bash
python get-mid.py
```
This will generate the final step list.

Paste the resulting steps into the `oss_steps` variable in: `wan/image2video_mdinfer_oss_stu.py`

#### Inference Command After Distillation

Replace the `generate-pi-i2v.py` in [Inference](#Inference) with `generate-pi-i2v-myinfer-oss-stu.py`







    
