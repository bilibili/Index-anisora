# -*- encoding: utf-8 -*-
# @File    :   finetune_roberta_copa.py
# @Time    :   2022/1/8
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
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
    keys = ["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"]
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
    tokens_1 = data_b['input_ids_1'].long()
    tokens_2 = data_b['input_ids_2'].long()
    tokens = torch.cat([tokens_1, tokens_2], dim=0)
    labels = data_b['label'].long()
    position_ids_1 = data_b['position_ids_1'].long()
    position_ids_2 = data_b['position_ids_2'].long()
    position_ids = torch.cat([position_ids_1, position_ids_2], dim=0)

    attention_mask_1 = data_b['attention_mask_1'][:, None, None, :].float()
    attention_mask_2 = data_b['attention_mask_2'][:, None, None, :].float()
    attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=0)

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, attention_mask, position_ids, (tokens!=1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask)
    bz = logits.shape[0] // 2
    logits = logits.squeeze(-1)[:,0].reshape(2, bz).permute(1, 0)

    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = logits.contiguous().float()
    loss = torch.nn.functional.cross_entropy(
        pred,
        labels
    )
    acc = (torch.argmax(pred, dim=1).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

def _encode(text):
    encoded_input = tokenizer(text, max_length=args.sample_length, padding='max_length')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

from sat.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        type = row['question']
        premise, choice1, choice2 = row['premise'], row['choice1'], row['choice2']
        premise = premise[:-1]
        choice1 = choice1[0].lower() + choice1[1:]
        choice2 = choice2[0].lower() + choice2[1:]
        if type=='cause':
            sentence1 = premise + ' because ' + choice1
            sentence2 = premise + ' because ' + choice2
        else:
            sentence1 = premise + ' so ' + choice1
            sentence2 = premise + ' so ' + choice2
            pass
        pack_1 = _encode(sentence1)
        pack_2 = _encode(sentence2)
        label = int(row['label'])
        return {
            'input_ids_1': np.array(pack_1['input_ids'], dtype=np.int64),
            'input_ids_2': np.array(pack_2['input_ids'], dtype=np.int64),
            'position_ids_1': np.array(pack_1['position_ids'], dtype=np.int64),
            'position_ids_2': np.array(pack_2['position_ids'], dtype=np.int64),
            'attention_mask_1': np.array(pack_1['attention_mask'], dtype=np.int64),
            'attention_mask_2': np.array(pack_2['attention_mask'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"], cache_dir='/dataset/fd5061f6/satDatasets', offline=True, transformer_name="copa_transformer")

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
