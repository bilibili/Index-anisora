##  🚀 Quick Started

### 1. Environment Set Up
```bash
cd reward
conda create -n AnimeReward python=3.10
conda activate AnimeReward
sh run_env.sh
```

### 2. Download Pretrained Weights

Please download our checkpoints from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/reward/weights) and put it in `./weights/`.

Please see [README.md](./character/README.md) from the character part for checkpoints download and data preparation.


### 3. Scoring for a single image-video case.

```bash
python reward_infer.py
```