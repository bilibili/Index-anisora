# -*- encoding: utf-8 -*-
'''
@File    :   chat_sat.py
@Time    :   2023/04/28 13:49:55
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any


import torch
from sat import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

from sat.generation.utils import timed_name, generate_continually

def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    '''GLM model, different from GPT.
    Args:
        seq: torch.IntTensor, [seq_len]
        mask_position: int, the position of the masked place.
        context_length: int, the length of context.
    Returns:
        tokens: torch.IntTensor, [1, seq_len]
        attention_mask: torch.FloatTensor, [1, seq_len, seq_len]
        position_ids: torch.IntTensor, [2, seq_len]
    '''
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    # 2D position ids
    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response

def chat(model, tokenizer, 
        query: str, history: List[Tuple[str, str]] = None, 
        max_length: int = 1024, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    if not history:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    # ---------------
    # tokenizer, this is an example of huggingface tokenizer.
    # input str, output['input_ids'] = tensor([[tokenized str, gmask, sop]])
    inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
    # ---------------
    # Next, we manually set the format to keep flexibility.
    mask_position = len(inputs) - 2
    context_length = len(inputs) - 1 # all before sop
    get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=context_length)
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    # ---------------
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id])
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)
    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=strategy
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    output_list = list(output)
    for i in range(len(output_list)):
        output = list(output_list[i])
        try:
            unfinished = output.index(-1)
        except ValueError:
            unfinished = len(output)
        if output[unfinished - 1] == tokenizer.eos_token_id:
            unfinished -= 1
        bog = output.index(tokenizer.bos_token_id)
        output_list[i] = output[:mask_position] + output[bog + 1:unfinished] + output[mask_position + 1:bog]
    # ---------------

    response = tokenizer.decode(output_list[0])
    response = process_response(response).split('答：')[-1].strip()
    history = history + [(query, response)]
    return response, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.95)
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained('chatglm-6b', args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    history = None
    with torch.no_grad():
        while True:
            query = input(">>> ")
            if query == "exit":
                break
            response, history = chat(model, tokenizer, query, history=history, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
            print(response.split('答：')[-1].strip())

            
