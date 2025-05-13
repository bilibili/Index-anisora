##  🚀 Quick Started

### 1. Environment Set Up
Download docker from [ascendhub](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a), the code is running based on 2024.rc3-x86.

```bash
cd anisoraV2_npu
pip3 install -r req.txt
pip3 install -r fastvideo/bili_space/requirements.txt
pip3 install -e .
pip3 install https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=0e59377b27823dda6d26528febb7ca06fc5b77816eaa58b4420cc8785e33d4ce
pip3 install torch-npu==2.4.0
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.19.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=7b2b46d3c757fa1fe7d08d48bdae6c9d97d82ace707474a79219c10991fee6ff
pip3 install diffusers-0.32.1-py3-none-any.whl
```
    
### 2. Download Pretrained Weights

Please download anisoraV2 checkpoints from [Huggingface] comming soon...

```bash
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
```
    
### 2. Data Preprocessing

1. Prepare video folder and prompt file (see `test_data/mp4root`,` test_data/train_data_prompts.txt`).
2. Modify input path, resolution, fps, and checkpoint root in the 4 feature extraction scripts under `preprocess/`.
3. Run the scripts to generate `.pt` feature files.
4. Organize training data following the format in `test_data/data.json`, place features in `test_data/test_pts/`

### 3. Training

```bash
bash scripts/finetune/finetune_wan.sh
```

### 4. Inference
```bash
bash scripts/inference/inference_wan.sh
```

Where,

    The --prompt file contains prompts and image paths for all cases (format: one line per case, image_path@@image_prompt), see test_data/

        An aesthetic score suffix can be fixed to 5.5, motion score is recommended between 1 and 2 (higher means more motion), adding "The video has no subtitles." means removing subtitles

    --image specifies the output folder

    --nproc_per_node is the number of GPUs (processes) used for distributed inference

    --ulysses_size must be a divisor of nproc_per_node. For example, if nproc_per_node=16 and ulysses_size=2, then every 2 GPUs infer one case, with 8 cases running in parallel. If you encounter a dim_size error, make sure that the dimension size is divisible by ulysses_size, or resize the image dimensions in advance

    --ckpt_dir is the root directory of model checkpoint

    --frame_num is the number of frames to infer, at 16fps, 49 frames equals about 3 seconds, must satisfy F=8x+1 (x∈Z)


