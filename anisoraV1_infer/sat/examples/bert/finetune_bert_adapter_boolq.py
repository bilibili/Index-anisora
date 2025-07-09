import os

import torch
import argparse
import numpy as np

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.model.official.bert_model import BertModel
from sat.model.mixins import MLPHeadMixin, AdapterMixin
from sat.model.base_model import BaseMixin
import torch.nn as nn
        

class AdapterModel(BertModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-12, **kwargs):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, **kwargs)
        self.del_mixin('bert-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('adapter', AdapterMixin(args.num_layers, args.hidden_size, args.adapter_hidden))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    def disable_untrainable_params(self):
        enable = ['layernorm', 'adapter', 'classification_head']
        for n, p in self.named_parameters():
            flag = False
            for e in enable:
                if e in n.lower():
                    flag = True
                    break
            if not flag:
                p.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-finetune', 'BERT finetune Configurations')
        # group.add_argument('--prefix_len', type=int, default=16)
        group.add_argument('--adapter_hidden', type=int, default=64)
        return super().add_model_specific_args(parser)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'token_type_ids', 'attention_mask', 'label']
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
    tokens = data_b['input_ids'].long()
    labels = data_b['label'].long()
    position_ids = data_b['position_ids'].long()
    token_type_ids = data_b['token_type_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    
    return tokens, labels, attention_mask, position_ids, token_type_ids, (tokens!=1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, token_type_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = logits.contiguous().float().squeeze(-1)[..., 0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

def _encode(text, text_pair):
    tokenizer = get_tokenizer()
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    seq_len = len(encoded_input['input_ids'])
    position_ids = torch.arange(seq_len)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids.numpy(), token_type_ids=encoded_input['token_type_ids'], attention_mask=encoded_input['attention_mask'])

from sat.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack, label = _encode(row['passage'], row['question']), int(row['label'])
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'token_type_ids': np.array(pack['token_type_ids'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "token_type_ids", "attention_mask", "label"], cache_dir=args.data_root, offline=False, transformer_name="boolq_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--data_root', type=str)
    py_parser.add_argument('--md_type', type=str)
    py_parser = AdapterModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    model, args = AdapterModel.from_pretrained(args.md_type, args)
    
    get_tokenizer(args)
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
