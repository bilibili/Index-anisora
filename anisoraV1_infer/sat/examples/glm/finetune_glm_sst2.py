# -*- encoding: utf-8 -*-
'''
@File    :   finetune_glm_sst2.py
@Time    :   2021/12/12 20:53:28
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

from sat.data_utils.datasets import TSVDataset
import torch
import argparse
import numpy as np

from sat import mpu, get_args, get_tokenizer
from sat.model.base_model import BaseModel, BaseMixin, non_conflict
from sat.training.deepspeed_training import training_main
from sat.data_utils import TSVDataset
from sat.model import GLMModel
from sat.model.transformer import standard_attention
from sat.model.mixins import MLPHeadMixin, PrefixTuningMixin

class ClassificationModel(GLMModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)
    
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
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['sentence'].long()
    labels = data_b['label'].long()
    batch_size, seq_length = tokens.size()
    
    position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)
    
    attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), device=tokens.device)

    attention_mask[...,:seq_length] -= (tokens==-1).view(batch_size, 1, 1, seq_length).float()
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    return tokens, labels, attention_mask, position_ids, (tokens!=-1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask)
    pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, 
        labels.float()
        )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    def process_fn(row):
        sentence, label = tokenizer._encode(row[0]), int(row[1])
        sentence = [tokenizer.get_command('ENC').Id] + sentence + [tokenizer.get_command('eos').Id]
        if len(sentence) >= args.sample_length:
            sentence = sentence[:args.sample_length]
        else:
            sentence.extend([-1] * (args.sample_length-len(sentence)))
        return {'sentence': np.array(sentence, dtype=np.int64), 'label': label}
    return TSVDataset(path, process_fn, with_heads=True)

if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=80)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
