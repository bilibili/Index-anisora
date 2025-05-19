##  🚀 Quick Started

### 1. Environment Setup

```bash
cd anisora_rl
conda create -n rl_infer python=3.10
conda activate rl_infer
pip install -r requirements.txt
cd SwissArmyTransformer-main
pip install -e .
cd ..
```

### 2. Download Pretrained Weights

Please download the text_encoder and VAE from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/CogVideoX_VAE_T5) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put them in `./sat/pretrained_models/`. 

Please download 5B_RL model weights from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/5B_RL) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put it in `./sat/ckpt/`.

### 3. Inference

```bash
cd sat
bash inference.sh 
```

