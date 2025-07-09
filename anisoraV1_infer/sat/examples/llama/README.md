# LLaMA v1 & v2

We support both llama v1 and v2.

You can run inference for llama model by:

```
python inference.py --mode inference --skip-init --fp16
python chat_sat.py
```

For large model, you need to use model parallel to inference:

```
torchrun --standalone --nnodes=1 --nproc-per-node=8 split_model.py
```

Here is the weight transformation script from [huggingface weight](https://huggingface.co/docs/transformers/main/model_doc/llama) and [meta-llama-2](https://huggingface.co/meta-llama):

```
python transform_param.py
```