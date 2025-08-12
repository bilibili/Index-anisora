# YOLOS

**YOLOS is adapted from https://github.com/hustvl/YOLOS**

## Pretrained weights

You can download pretrained weights from YOLOS repo, and then transform it into `sat` using `transform_param.py`.

```bash
python transform_param.py
```

## Inference

We adapt [VisualizeAttention.ipynb](https://github.com/hustvl/YOLOS/blob/main/VisualizeAttention.ipynb) here as `inference_yolos.py`.

## Training

```bash
bash scripts/finetune_coco.sh /data/qingsong/pretrain
```

For now, we haven't support customized evaluator. Therefore, I just report loss as metric.

Moreover, you may need to change coco_path in the train_yolos_coco.py to your own data path.