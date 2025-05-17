##  🚀 Quick Started

### 1. Environment Set Up

```bash
cd data_pipeline
conda create -n data python=3.10
conda activate data
pip install -r requirements.txt
```

### 2. Download Checkpoints

Please download the checkpoints from [HuggingFace](https://huggingface.co/IndexTeam/Index-anisora/tree/main/data_pipeline) and put them in the `./weights/`.

### 3. Start inference

```bash
python infer_single.py --in_vid sample/918195104_1325613009-Scene-008.mp4
```
