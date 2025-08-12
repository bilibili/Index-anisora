# swiss-bert

We transform parameters of bert-base-uncased from huggingface to swiss by `transform_param.py`.

**Replace /data/qingsong/pretrain to your model root and /data/qingsong/dataset to your data root!**

## Inference

```bash
bash scripts/inference_bert.sh /data/qingsong/pretrain
```

## Finetune

Finetune bert-base-uncased:

```bash
bash scripts/finetune_boolq.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iter 1000 val acc: 0.755

Finetune bert with adapter:

```bash
bash scripts/finetune_adapter_boolq.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iter 1000 val acc: 0.636

Finetune bert by distillation:

* First, train a bert large with scripts/finetune_boolq.sh
* Then, distill by the following command

```bash
bash scripts/finetune_distill_boolq.sh /data/qingsong/pretrain /data/qingsong/dataset checkpoints/finetune-bert-large-uncased-boolq06-21-05-18
```

You should replace `checkpoints/finetune-bert-large-uncased-boolq06-21-05-18` to your own teacher model.