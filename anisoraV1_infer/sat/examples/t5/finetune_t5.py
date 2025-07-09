# -*- encoding: utf-8 -*-
'''
@File    :   finetune_t5.py
@Time    :   2021/12/11 02:39:13
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
import argparse
import numpy as np

from sat import mpu, get_args, get_tokenizer
from sat.model.base_model import BaseModel, BaseMixin
from sat.training.deepspeed_training import training_main
from sat.data_utils import TSVDataset
from sat.model import T5Model

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['sentence', 'label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data = data[0].to('cuda')
    attention_mask = torch.ones((1, data.shape[-1], data.shape[-1]), device=data.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    return data, torch.arange(data.shape[-1], device=data.device), attention_mask
    


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    input_ids, position_ids, mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    # Forward model.
    
    enc, logits, *mems = model(
        enc_input_ids=input_ids,
        dec_input_ids=input_ids, 
        enc_position_ids=position_ids,
        dec_position_ids=position_ids,
        dec_attention_mask=mask)
    # logits, *mems = model(tokens, position_ids, attention_mask)
    loss = logits.mean()
    return loss, {}

def create_dataset_function(path, args):
    
    return torch.utils.data.TensorDataset(
        torch.ones(100000, 20, dtype=torch.long)
    )

if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=80)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--cache-dir', type=str, default='/root/some_cache',
                           help='hf cache')
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    training_main(args, model_cls=T5Model, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
