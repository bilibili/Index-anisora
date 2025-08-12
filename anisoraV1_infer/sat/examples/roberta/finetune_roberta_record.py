# -*- encoding: utf-8 -*-
# @File    :   finetune_roberta_msc.py
# @Time    :   2022/1/11
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import os

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from functools import partial
from sat import mpu, get_args
from sat.training.deepspeed_training import training_main
from sat.model.official.roberta_model import RobertaModel
from sat.model.mixins import PrefixTuningMixin, MLPHeadMixin, BaseMixin

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ["pos_tokens", "neg_tokens"]
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    pos_tokens_2dim = data_b['pos_tokens'].long()
    neg_tokens_2dim = data_b['neg_tokens'].long()
    pos_tokens = pos_tokens_2dim.reshape([-1, args.max_positive, args.sample_length])
    neg_tokens = neg_tokens_2dim.reshape([-1, args.max_negative, args.sample_length])

    padding_idx = 1
    pos_tokens_position_ids = create_position_ids_from_input_ids(pos_tokens.reshape([pos_tokens.shape[0] * pos_tokens.shape[1], -1]), 1, 0).reshape(pos_tokens.shape)
    neg_tokens_position_ids = create_position_ids_from_input_ids(neg_tokens.reshape([neg_tokens.shape[0] * neg_tokens.shape[1], -1]), 1, 0).reshape(neg_tokens.shape)
    pos_tokens_attention_mask = (torch.ones_like(pos_tokens, device=pos_tokens.device) * pos_tokens_position_ids.ne(padding_idx))[:, :, None, None, :].float()
    neg_tokens_attention_mask = (torch.ones_like(neg_tokens, device=neg_tokens.device) * neg_tokens_position_ids.ne(padding_idx))[:, :, None, None, :].float()


    # Convert
    if args.fp16:
        pos_tokens_attention_mask = pos_tokens_attention_mask.half()
        neg_tokens_attention_mask = neg_tokens_attention_mask.half()
    return pos_tokens, pos_tokens_position_ids, pos_tokens_attention_mask, neg_tokens, neg_tokens_position_ids, neg_tokens_attention_mask

def get_masked_input(tokens, mask):
    masked_tokens = tokens.clone()
    masked_tokens[mask] = 50264
    return masked_tokens

def get_lprobs(model, tokens, mask, position_ids, attention_mask):
    logits, *mems = model(get_masked_input(tokens, mask), position_ids, attention_mask)
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
    scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
    mask = mask.type_as(scores)
    scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
    return scores

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    pos_tokens, pos_tokens_position_ids, pos_tokens_attention_mask, neg_tokens, neg_tokens_position_ids, neg_tokens_attention_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    loss = 0.0
    ncorrect, nqueries = 0, 0
    for i in range(pos_tokens.shape[0]):
        pos_tokens_now = pos_tokens[i]
        neg_tokens_now = neg_tokens[i]
        pos_num = pos_tokens_now.shape[0]
        neg_num = neg_tokens_now.shape[0]
        for j in range(pos_tokens_now.shape[0]):
            if pos_tokens_now[j][0] == -1:
                pos_tokens_now = pos_tokens_now[:j]
                pos_num = j
                break
        for j in range(neg_tokens_now.shape[0]):
            if neg_tokens_now[j][0] == -1:
                neg_tokens_now = neg_tokens_now[:j]
                neg_num = j
                break
        logits_pos, *mems = model(pos_tokens_now, pos_tokens_position_ids[i][:pos_num], pos_tokens_attention_mask[i][:pos_num])
        logits_neg, *mems = model(neg_tokens_now, neg_tokens_position_ids[i][:neg_num], neg_tokens_attention_mask[i][:neg_num])
        for j in range(pos_tokens_now.shape[0]):
            for k in range(neg_tokens_now.shape[0]):
                loss +=  F.margin_ranking_loss(logits_pos[j][0], logits_neg[k][0], torch.ones_like(logits_pos[j][0], device=pos_tokens_now.device).sign(), 1)
        if torch.max(logits_pos[:,0]) > torch.max(logits_neg[:,0]):
            ncorrect += 1
        nqueries += 1

    acc = torch.tensor(ncorrect/nqueries, device=pos_tokens.device)
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
        passage = row['passage']
        query = row['query']
        entities = row['entities']
        pos = row['answers']

        #clean data
        neg = [entity for entity in entities if entity not in pos]

        pos_tokens = []
        neg_tokens = []
        for word in pos:
            new_query = query.replace('@placeholder', word)
            pos_tokens.append(_encode(passage, new_query)['input_ids'])
        for word in neg:
            new_query = query.replace('@placeholder', word)
            neg_tokens.append(_encode(passage, new_query)['input_ids'])

        if len(neg_tokens) > args.max_negative:
            print("truncate negative")
            neg_tokens = neg_tokens[:args.max_negative]
        if len(pos_tokens) > args.max_positive:
            print("truncate positive")
            pos_tokens = pos_tokens[:args.max_positive]
        res_positive = args.max_positive - len(pos_tokens)
        res_negative = args.max_negative - len(neg_tokens)
        for i in range(res_positive):
            pos_tokens.append([-1] * args.sample_length)
        for i in range(res_negative):
            neg_tokens.append([-1] * args.sample_length)
        pos_tokens = np.stack(pos_tokens)
        neg_tokens = np.stack(neg_tokens)
        pos_tokens = pos_tokens.reshape(-1)
        neg_tokens = neg_tokens.reshape(-1)
        return {
            'pos_tokens': pos_tokens,
            'neg_tokens': neg_tokens,
        }

    return load_hf_dataset(path, process_fn, columns = ["pos_tokens", "neg_tokens"], cache_dir='/dataset/fd5061f6/satDatasets', offline=True, transformer_name="record_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--max_positive', type=int, default=5)
    py_parser.add_argument('--max_negative', type=int, default=20)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
