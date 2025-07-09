# ChatGLM3

This folder provides [ChatGLM3-6B](https://github.com/THUDM/ChatGLM3) model in SAT.

If you want to inference or chat with ChatGLM3-6B:

```
python chat_sat.py
```

Models will be downloaded and cached automatically into `~/.sat_models`.

For finetuning ChatGLM3-6B, you can refer to the [../chatglm2](../chatglm2) folder.

Here is the parameter transformation script (from huggingface/chatglm3-6b\* to sat/chatglm3-6b\*):

```
python transform_param.py
```