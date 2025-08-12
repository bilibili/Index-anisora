import os
import torch
import argparse
from sat import get_args
from finetune_chatglm2 import FineTuneModel

py_parser = argparse.ArgumentParser(add_help=False)
py_parser.add_argument('--ckpt_path', type=str)
py_parser = FineTuneModel.add_model_specific_args(py_parser)
known, args_list = py_parser.parse_known_args()
args = get_args(args_list)
args = argparse.Namespace(**vars(args), **vars(known))

model_path = args.ckpt_path
from chat_model import ChatModel
model, args = ChatModel.from_pretrained(model_path, args, FineTuneModel)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = model.eval().cuda()
while True:
    summary = input("输入提示语：")
    response, history = model.chat(tokenizer, summary)
    print(response)