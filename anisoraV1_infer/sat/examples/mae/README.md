# MAE

## Pretrained weights

You can download pretrained weights from [facebookresearch/mae](https://github.com/facebookresearch/mae/issues/8).

Then use `transform_param.py` to transform weights into `sat`.

```bash
python transform_param.py
```

## Inference

`inference_mae.ipynb`

## Fine-tune Encoder

```bash
bash scripts/finetune_cifar10.sh
```

* mae-base: iteration 1000:
    * 224x224 acc 0.9739
    * 384x384 acc 0.9470

The results show that fixed position embeddings do not generalize well to larger resolution.