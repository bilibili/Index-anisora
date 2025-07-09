from dotenv import load_dotenv
load_dotenv()

import os
import torch
import argparse
from sat import get_args, AutoModel

args = get_args()

model_type = 'chatglm-6b'
model, args = AutoModel.from_pretrained(model_type, args)
device = model.parameters().__next__().device
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

model = model.eval()
with torch.no_grad():
    text = ["This is a piece of text.", "Another piece of text."]
    """
    It is noteworthy that for inference, if you want to use batch input with variable seq_len,
    you need to feed all outputs of tokenizer into model, including input_ids, position_ids, attention_mask.
    Feeding a single input_ids to model only works for training, not inference.
    This is because for generation or inference, tokens are left padded.
    """
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input = {k:v.to(device) for k, v in encoded_input.items()}
    dst_output = model(**encoded_input)
    output = dst_output[0].cpu()
    print(output)

breakpoint()