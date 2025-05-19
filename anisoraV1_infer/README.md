##  🚀 Quick Started

### 1. Environment Setup

```bash
cd anisoraV1_infer 
conda create -n ani_infer python=3.10
conda activate ani_infer
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

Please download the text_encoder and VAE from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/CogVideoX_VAE_T5) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put them in `./pretrained_models/`. 

Please download 5B model weights from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/5B) or [ModelScope](https://modelscope.cn/models/bilibili-index/Index-anisora/files) and put it in `./ckpt/`.

### 3. Inference

For, A100 , you can set `offload=0`:
```bash
offload=0 python demo.py --base configs/cogvideox/cogvideox_5b_720_169_2.yaml
```

For, 4x4090, you must set `offload=1`:
```bash
offload=1 python demo.py --base configs/cogvideox/cogvideox_5b_720_169_2.yaml 
```

## 📁 System Notes

- **Supports up to SP4**
- **2 × RTX 4090** runs **OOM** during decoding
- **RTX 4090** only supports up to **720×1088** resolution  
  - Therefore, for testing:  
    - 4090 uses **720×1088** 
    - A800 uses standard **720×1280**

---

## 🍎 Performance Results

| Configuration | Time (seconds) |
|---------------|----------------|
| 4×RTX 4090     | 341            |
| 4×A800         | 224            |
| 2×A800         | 356            |
| 1×A800         | 609            |


