## training
Data will be automatically downloaded from hugging face. 

### Training Mantis based on LLaMA3 with CLIP/SigLIP encoder:
- Pretrain Mantis-LLaMA3 Multimodal projector on pretrain data (Stage 1):
```bash
bash scripts/pretrain_mllava.sh
```

- Fine-tune the pretrained Mantis-LLaMA3 on Mantis-Instruct (Stage 2):
```bash
bash scripts/train_mllava.sh
```

### Training Mantis based on Fuyu-8B:
- Fine-tune Fuyu-8B on Mantis-Instruct to get Mantis-Fuyu:
```bash
bash scripts/train_fuyu.sh
```

To accelerate the training with fuyu, please also install the flash-attention packages for some submodules in the model. Commands:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
cd csrc
cd layer_norm && pip install . && cd ..
cd rotary && pip install . && cd ..
cd fused_dense_lib && pip install . && cd ..
cd xentropy && pip install . && cd ..
cd ..
rm -rf flash-attention
```

Please remember to modify the DATA_CONFIG_FILE in the bash script. You can try on a small subset of Mantis-Instruct (5K examples, Multi-VQA) by setting it to be `DATA_CONFIG_FILE="./data_configs/train_config_debug.yaml"`, which help you debug for your local environment.

Our official Mantis-Instruct data_config file is `./data_configs/train_config.yaml`.