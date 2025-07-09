# swiss-roberta

We transform parameters of [roberta-base](https://huggingface.co/roberta-base) and [roberta-large](https://huggingface.co/roberta-large) from huggingface to swiss by `transform_param.py`.

## Inference

```bash
bash scripts/inference_roberta.sh /path/to/model
```

## Finetune

```bash
bash scripts/finetune_boolq.sh base
bash scripts/finetune_boolq.sh large
```

### Performace

|                                 | Boolq | CB   | COPA | MultiRC | ReCoRD | RTE  | WSC  | WIC  |
|---------------------------------| ----- | ---- | ---- | ------- |--------| ---- | ---- | ---- |
| roberta-large(Liu et al., 2019) | 86.9  | 98.2 | 94.0 | 85.7    | 89.0   | 86.6 | 89.0 | 75.6 |
| roberta-large(ours)             | 85.6  | 97.2 | 93.7 | 85.2    | 90.7   | 87.1 | 84.6 | 74.2 |
| roberta-large + prefix tuning   | 86.8  | 96.4 | 95.8 | 84.7    | 90.8   | 87.1 | 84.6 | 75.8 |

[^1]: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

