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
        # self.del_mixin('roberta-final')
        # self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ["label", "query_tokens", "query_mask", "cand_tokens", "cand_mask", "label"]
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
    query_tokens = data_b['query_tokens'].long()
    query_mask = data_b['query_mask'].long()
    cand_tokens = data_b['cand_tokens'].long()
    cand_mask = data_b['cand_mask'].long()
    labels = data_b['label'].long()

    query_mask = query_mask.bool()
    cand_mask = cand_mask.bool()

    cand_tokens = cand_tokens.reshape([query_tokens.shape[0], -1, query_tokens.shape[1]])
    cand_mask = cand_mask.reshape([query_mask.shape[0], -1, query_mask.shape[1]])
    padding_idx = 1
    query_position_ids = create_position_ids_from_input_ids(query_tokens, 1, 0)
    query_attention_mask = (torch.ones_like(query_position_ids, device=query_tokens.device) * query_position_ids.ne(padding_idx))[:,None,None,:].float()
    cand_tokens_position_ids = create_position_ids_from_input_ids(cand_tokens.reshape([cand_tokens.shape[0]*cand_tokens.shape[1],-1]), 1, 0).reshape(cand_tokens.shape)
    cand_attention_mask = (torch.ones_like(cand_tokens_position_ids, device=query_tokens.device) * cand_tokens_position_ids.ne(padding_idx))[:,:,None,None,:].float()

    # Convert
    if args.fp16:
        query_attention_mask = query_attention_mask.half()
        cand_attention_mask = cand_attention_mask.half()
    return query_tokens, query_mask, query_attention_mask, query_position_ids, cand_tokens, cand_mask, cand_attention_mask, cand_tokens_position_ids, labels

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
    query_tokens, query_mask, query_attention_mask, query_position_ids, cand_tokens, cand_mask, cand_attention_mask, cand_tokens_position_ids, labels = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    loss, nloss = 0.0, 0
    ncorrect, nqueries = 0, 0
    for i in range(labels.shape[0]):
        label = labels[i]
        query_lprobs = get_lprobs(
            model,
            query_tokens[i].unsqueeze(0),
            query_mask[i].unsqueeze(0),
            query_position_ids[i].unsqueeze(0),
            query_attention_mask[i].unsqueeze(0),
        )
        cand_tokens_now = cand_tokens[i]
        cand_mask_now = cand_mask[i]
        cand_position_ids_now = cand_tokens_position_ids[i]
        cand_attention_mask_now = cand_attention_mask[i]
        for j in range(args.max_cand_len):
            if cand_tokens_now[j][0] == -1:
                cand_tokens_now = cand_tokens_now[:j]
                cand_mask_now = cand_mask_now[:j]
                cand_position_ids_now = cand_position_ids_now[:j]
                cand_attention_mask_now = cand_attention_mask_now[:j]
                break
        cand_lprobs = get_lprobs(
            model,
            cand_tokens_now,
            cand_mask_now,
            cand_position_ids_now,
            cand_attention_mask_now,
        )
        pred = (query_lprobs >= cand_lprobs).all().item()
        ncorrect += 1 if pred == label else 0
        nqueries += 1
        if label:
            nloss += 1
            loss += F.cross_entropy(
                torch.cat([query_lprobs, cand_lprobs]).unsqueeze(0),
                query_lprobs.new([0]).long(),
            )

    if nloss == 0:
        loss = torch.tensor(0.0, requires_grad=True, device=query_tokens.device)
    acc = torch.tensor(ncorrect/nqueries, device=query_tokens.device)
    return loss, {'acc': acc}

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

def _encode(text):
    encoded_input = tokenizer(text, max_length=args.sample_length, padding='max_length', truncation='only_first')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

import spacy

from sat.data_utils import load_hf_dataset
import en_core_web_sm

def create_dataset_function(path, args):
    def process_fn(row):
        nlp = en_core_web_sm.load()
        pad = lambda a: a[0:args.sample_length] if len(a) > args.sample_length else a + [1] * (args.sample_length-len(a))
        text = row['text']
        query = row['span1_text']
        if query[-1]=='.' or query[-1] == ',':
            query = query[:-1]
        sentence = nlp(text)
        encoded_text = tokenizer(text)['input_ids']

        start2 = row['text'].find(row['span2_text'])
        pron_index = tokenizer(row['text'][:start2])['input_ids'].__len__() - 2
        encoded_query = tokenizer(query)['input_ids'][1:-1]
        prefix = encoded_text[:pron_index]
        suffix = encoded_text[pron_index+1 :]
        query_tokens = pad(prefix + encoded_query + suffix)
        query_tokens = np.array(query_tokens)
        query_mask = np.zeros_like(query_tokens)
        query_mask[len(prefix):len(prefix)+len(encoded_query)] = 1

        #extend chunks
        noun_chunks = {(nc.start, nc.end) for nc in sentence.noun_chunks}
        np_start, cur_np = 0, "NONE"
        for i, token in enumerate(sentence):
            np_type = token.pos_ if token.pos_ in {"NOUN", "PROPN"} else "NONE"
            if np_type != cur_np:
                if cur_np != "NONE":
                    noun_chunks.add((np_start, i))
                if np_type != "NONE":
                    np_start = i
                cur_np = np_type
        if cur_np != "NONE":
            noun_chunks.add((np_start, len(sentence)))
        noun_chunks = [sentence[s:e] for (s, e) in sorted(noun_chunks)]

        #exclude pron and query
        noun_chunks = [
            nc
            for nc in noun_chunks
            if (nc.lemma_ != "-PRON-" and not all(tok.pos_ == "PRON" for tok in nc))
        ]
        excl_txt = [query.lower()]
        filtered_chunks = []
        for chunk in noun_chunks:
            lower_chunk = chunk.text.lower()
            found = False
            for excl in excl_txt:
                if (lower_chunk in excl or excl in lower_chunk) or lower_chunk == excl:
                    found = True
                    break
            if not found:
                filtered_chunks.append(chunk)

        noun_chunks = filtered_chunks

        #calc cand_tokens
        cand_token_list = []
        cand_mask_list = []
        for cand_span in noun_chunks:
            encoded_cand = tokenizer(cand_span.text)['input_ids'][1:-1]
            cand_tokens = pad(prefix + encoded_cand + suffix)
            cand_tokens = np.array(cand_tokens)
            cand_mask = np.zeros_like(cand_tokens)
            cand_mask[len(prefix):len(prefix)+len(encoded_cand)] = 1
            cand_token_list.append(cand_tokens)
            cand_mask_list.append(cand_mask)

        if len(cand_token_list) > args.max_cand_len:
            cand_token_list = cand_token_list[:args.max_cand_len]
            cand_mask_list = cand_mask_list[:args.max_cand_len]
        for i in range(args.max_cand_len - len(cand_token_list)):
            cand_token_list.append([-1] * args.sample_length)
            cand_mask_list.append([0] * args.sample_length)
        cand_tokens = np.stack(cand_token_list)
        cand_mask = np.stack(cand_mask_list)
        cand_tokens = cand_tokens.reshape([-1])
        cand_mask = cand_mask.reshape([-1])

        label = int(row['label'])
        return {
            'label': label,
            'query_tokens': query_tokens,
            'query_mask' : query_mask,
            'cand_tokens': cand_tokens,
            'cand_mask': cand_mask
        }

    return load_hf_dataset(path, process_fn, columns = ["label", "query_tokens", "query_mask", "cand_tokens", "cand_mask"], cache_dir='/dataset/fd5061f6/satDatasets', offline=True, transformer_name="wsc_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--max_cand_len', type=int, default=20)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
