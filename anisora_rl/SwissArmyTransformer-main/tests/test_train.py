import os
import sys
import math
import random
import torch
import argparse
import numpy as np

from sat import mpu, get_args, get_tokenizer
from sat.model.base_model import BaseModel
from sat.training.deepspeed_training import training_main
from sat.data_utils import BinaryDataset
from torch.utils.data import TensorDataset, DataLoader


def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
        attention_mask.tril_()
        attention_mask.unsqueeze_(1)
        
    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask']
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
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()

    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    attention_mask = None        

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        args=args
        )
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)  

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    return loss, {}


class FakeDataset(TensorDataset):
    def __init__(self, process_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_fn = process_fn

    def __getitem__(self, idx):
        return self.process_fn(super().__getitem__(idx))

def create_dataset_function(path, args):
    num_samples = 100000
    num_features = 5

    x = torch.randint(0, 100, (num_samples, num_features))

    def process_fn(row):
        return {'text': row[0], 
            'loss_mask': np.array([1]*len(row[0]))
            }
    return FakeDataset(process_fn, x)

if __name__ == '__main__':    
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list + '--train-data fake --num-workers 0'.split())
    args = argparse.Namespace(**vars(args), **vars(known))
    
    training_main(args, model_cls=BaseModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
