## 🚀 Quick Start

## 1. Docker Image:

download docker from [there]().

## 2. Download Pretrained Models:

```bash
cd anisoraV1_train_npu 
download ./sat/5b_tool_models/ from our huggingface proj
```

## 3. Download Environment Packages:

download env_file from [there]() and put them at `./env_file/`. 

## 4. Prepare Training Data:

Please construct the json file for your dataset following the format of [data.json](./sat/demo_data/data.json). 
For VAE feature extraction, please refer to the [cli_vae_demo.py](https://github.com/THUDM/CogVideo/blob/main/inference/cli_vae_demo.py) in official CogVideoX repository.


## 5. Training

```bash
bash ./script/npu_run.sh
```


