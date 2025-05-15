## 🚀 Quick Start

### 1. Environment Set Up

```bash
cd anisoraV1_train_gpu
conda create -n anisoraV1_gpu python=3.10
conda activate anisoraV1_gpu
pip install -r requirements.txt
```

## 2. Download Pretrained Models:

```bash
cd anisoraV1_train_gpu 
download ./sat/5b_tool_models/ from our huggingface proj
```

## 3. Prepare Training Data:

Please construct the json file for your dataset following the format of [data.json](./sat/demo_data/data.json). 
For VAE feature extraction, please refer to the [cli_vae_demo.py](https://github.com/THUDM/CogVideo/blob/main/inference/cli_vae_demo.py) in official CogVideoX repository.


## 4. Training

```bash
bash ./script/gpu_run.sh
```


