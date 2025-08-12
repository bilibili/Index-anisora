# CaiT

CaiT is different from original ViT in the following aspects:

1. Encoder-Decoder architecture
2. For encoder part, they use LayerScale and TalkingHead attention without CLS token.
3. For decoder part, they use LayerScale with CLS token only.

## Pretrained weights

You can download pretrained weights from [CaiT in deit](https://github.com/facebookresearch/deit/blob/main/README_cait.md) and set configuration referring to [CaiT in timm](https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/models/cait.py#L329).

Then use `transform_param.py` to transform weights into `sat`.

```bash
python transform_param.py
```

## Usage

You can use the model similar to [ViTModel](../vit).

```bash
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset
```

val acc at iter 1000: 0.9834