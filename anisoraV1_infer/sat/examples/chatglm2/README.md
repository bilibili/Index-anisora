# ChatGLM2

This folder provides [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) model in SAT.

If you want to inference or chat with ChatGLM2-6B:

```
python chat_sat.py
```

If you want to chat with ChatGLM2-6B with huggingface generation instead of sat (legacy):

```
python chat.py --mode inference --fp16 --skip-init
```

Models will be downloaded and cached automatically into `~/.sat_models`.

For finetuning ChatGLM2-6B, we adapt the [official finetune code](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning) to SAT, and provide two parameter-efficient tuning methods, i.e., ptuning and lora.

```
bash scripts/finetune_adgen_ptuning.sh
bash scripts/finetune_adgen_lora.sh
python inference_adgen.py --mode inference --skip-init --fp16 --ckpt_path checkpoints/finetune-chatglm2-6b-adgen-07-06-14-29/
```

Here is the parameter transformation script (from huggingface/chatglm2-6b to sat/chatglm2-6b):

```
python transform_param.py
```