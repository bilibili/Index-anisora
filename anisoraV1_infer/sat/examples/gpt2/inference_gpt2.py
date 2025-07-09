import os
import torch
import argparse
from sat import get_args, AutoModel

args = get_args()

model_type = 'gpt2'
model, args = AutoModel.from_pretrained(model_type, args)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2 = GPT2LMHeadModel.from_pretrained(model_type)
tokenizer = GPT2Tokenizer.from_pretrained(model_type)

gpt2.eval()
model.eval()
with torch.no_grad():
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt')
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    print(position_ids)
    hugging_output = gpt2(**encoded_input).logits
    model.to('cuda:0')
    attention_mask = encoded_input['attention_mask'].unsqueeze(1).repeat_interleave(encoded_input['attention_mask'].shape[-1], 1).tril().unsqueeze(1)
    dst_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=attention_mask.cuda())
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())
