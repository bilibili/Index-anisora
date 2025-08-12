# -*- encoding: utf-8 -*-
# @File    :   finetune_roberta_wic.py
# @Time    :   2022/1/10
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import os

import torch
import argparse
import numpy as np

from sat import mpu, get_args
from sat.training.deepspeed_training import training_main
from sat.model.official.roberta_model import RobertaModel
from sat.model.mixins import PrefixTuningMixin, MLPHeadMixin, BaseMixin

class WIC_MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005):
        super().__init__()
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for sz in output_sizes:
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        cls_embedding = logits[:,0] #32 * hidden
        logits = logits.reshape([-1, logits.shape[-1]])
        pos1_embedding = logits[kw_args['pos1']] #32 * hidden
        pos2_embedding = logits[kw_args['pos2']]
        logits = torch.cat([cls_embedding, pos1_embedding, pos2_embedding], dim=-1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', WIC_MLPHeadMixin(args.hidden_size * 3, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'pos1', 'pos2']
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
    attention_mask = data_b['attention_mask'][:, None, None, :].float()
    pos1 = data_b['pos1'].long()
    pos2 = data_b['pos2'].long()
    bz = tokens.shape[0]
    for i in range(bz):
        pos1[i] += i * tokens.shape[1]
        pos2[i] += i * tokens.shape[1]

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, attention_mask, position_ids, (tokens!=1), pos1, pos2


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask, pos1, pos2 = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask, pos1=pos1, pos2=pos2)
    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = logits.contiguous().float().squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

def _encode(text, text_pair):
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

from sat.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack = _encode(row['sentence1'], row['sentence2'])
        label = int(row['label'])
        start1, end1 = int(row['start1']), int(row['end1'])
        start2, end2 = int(row['start2']), int(row['end2'])
        pos1 = tokenizer(row['sentence1'][:start1])['input_ids'].__len__() - 2
        pos2 = tokenizer(row['sentence2'][:start2])['input_ids'].__len__() - 2 + tokenizer(row['sentence1'])['input_ids'].__len__()
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'label': label,
            'pos1': pos1,
            'pos2': pos2
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "attention_mask", "label", "pos1", "pos2"], cache_dir='/dataset/fd5061f6/satDatasets', offline=False, transformer_name="wic_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
