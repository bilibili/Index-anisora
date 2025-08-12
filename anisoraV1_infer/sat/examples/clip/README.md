# CLIP

## Pretrained weights

You can download pretrained weights from [huggingface:openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

Then use `transform_param.py` to transform weights into `sat`.

```bash
python transform_param.py
```

## Inference

`inference_clip.ipynb`

or

```bash
bash scripts/inference_clip.sh /data/qingsong/pretrain
```

## Fine-tune

Here is an example of fine-tuning image encoder for CIFAR-10.

```bash
bash scripts/finetune_cifar10.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iteration 1000 validation acc: 0.9402

Here is an example of fine-tuning CIFAR-10 by prompt.

```bash
bash scripts/finetune_cifar10wp.sh /data/qingsong/pretrain /data/qingsong/dataset
```

* iteration 100 validation acc: 0.9568
* iteration 1000 validation acc: 0.9478

It's interesting that the accuracy going upward first, then dropping down, then going upward again.