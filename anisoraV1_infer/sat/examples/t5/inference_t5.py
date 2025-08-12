# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial

from sat import mpu, get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed

from sat.model import T5Model
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity, get_masks_and_position_ids_default
from sat.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from sat.generation.utils import timed_name, generate_continually
from sat.training.deepspeed_training import setup_model_and_optimizer



def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model 
    model = T5Model(args)
    # model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_checkpoint(model, args)
    set_random_seed(args.seed)
    model.eval()

    # test correctness
    input_ids = tokenizer.EncodeAsIds("The <extra_id_0> walks in <extra_id_1> park").tokenization
    input_ids = input_ids + [tokenizer.get_command("eos").Id]
    input_ids = torch.tensor(input_ids, device='cuda', dtype=torch.long)
    decoder_input_ids = tokenizer.EncodeAsIds('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>').tokenization
    decoder_input_ids = decoder_input_ids + [tokenizer.get_command("eos").Id]
    decoder_input_ids = torch.tensor(decoder_input_ids, device='cuda', dtype=torch.long)

    input_ids, _mask, enc_position_ids = get_masks_and_position_ids_default(input_ids)
    
    decoder_input_ids, dec_attention_mask, dec_position_ids = get_masks_and_position_ids_default(decoder_input_ids)
    
    encoder_outputs, decoder_outputs, *mems = model(
        enc_input_ids=input_ids,
        dec_input_ids=decoder_input_ids, 
        dec_attention_mask=dec_attention_mask
    )
    breakpoint()
    

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy',
                           help='type name of sampling strategy')
    py_parser.add_argument('--cache-dir', type=str, default='/root/some_cache',
                           help='hf cache')
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    with torch.no_grad():
        main(args)
