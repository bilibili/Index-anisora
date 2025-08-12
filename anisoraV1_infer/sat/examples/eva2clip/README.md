# EVA2-CLIP

This folder contains [EVA2-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) model in SAT format. The model architecture is also used in [CogVLM](https://github.com/THUDM/CogVLM/blob/main/utils/models/eva_clip_model.py).

We provide transformation code from official EVA to SAT. You can get official EVA code [here](https://github.com/baaivision/EVA/tree/master/EVA-CLIP/rei/eva_clip).

Copy the `eva_clip` folder here. Then run:

```
python transform_param.py
```