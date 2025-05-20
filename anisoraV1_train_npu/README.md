## 🚀 Quick Start

## 1. Docker Image:

Download docker from [ascendhub](https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a), the code is running based on 2024.rc3-x86.

## 2. Download Pretrained Models:

Please download the text_encoder and vae from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/CogVideoX_VAE_T5) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put them in `./sat/5b_tool_models/`. 

## 3. Prepare Training Data:

Please construct the json file for your dataset following the format of [data.json](./sat/demo_data/data.json). 
For VAE feature extraction, please refer to the [cli_vae_demo.py](https://github.com/THUDM/CogVideo/blob/main/inference/cli_vae_demo.py) in official CogVideoX repository.


## 4. Training

```bash
bash ./script/npu_run.sh
```


