# DeiT

DeiT shares the same architecture with ViTModel. Therefore, you can use `transform_param.py` to transform pretrained DeiT into `sat`, and then use it in a same way as in [examples/vit](../vit).

For now, we support three deit types, i.e., deit-tiny, deit-small and deit-base.

```bash
python transform_param.py --model tiny
python transform_param.py --model small
python transform_param.py --model base
```

For more model types, you may refer to [deit in timm](https://github.com/facebookresearch/deit/blob/main/models.py#L63).

## Finetune

```bash
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset tiny
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset small
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset base
```

cifar10 validation acc:

* tiny: 0.9584
* small: 0.9731
* base: 0.9809