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

Download VAE, text_encoder, and 5B_rl model weights from HuggingFace.

### 3. Inference

```bash
cd sat
bash inference.sh 
```

