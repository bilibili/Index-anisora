# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2021/10/06 17:58:04
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from sat.helpers import print_rank0


def get_tokenizer(args=None, *, tokenizer_type=None, outer_tokenizer=None):
    '''
        If you're using outer_tokenizer, call `get_tokenizer(args, outer_tokenizer)`
        before `training_main`.
    '''
    if outer_tokenizer is not None: # set 1
        get_tokenizer.tokenizer = outer_tokenizer
        get_tokenizer.tokenizer_type = 'outer_tokenizer'
        print_rank0('> Set tokenizer as an outer_tokenizer! Now you can get_tokenizer() everywhere.')
        return outer_tokenizer
    if tokenizer_type is None:
        if args is None:
            assert hasattr(get_tokenizer, 'tokenizer'), 'Never set tokenizer.'
            return get_tokenizer.tokenizer
        tokenizer_type = args.tokenizer_type

    # find the tokenizer via tokenizer_type!
    if hasattr(get_tokenizer, 'tokenizer_type') and \
        tokenizer_type == get_tokenizer.tokenizer_type:  # the same as last
        return get_tokenizer.tokenizer

    get_tokenizer.tokenizer_type = tokenizer_type
    # load the tokenizer according to tokenizer_type
    if tokenizer_type.startswith('cogview'): # or cogview_ICE
        from .cogview import UnifiedTokenizer
        get_tokenizer.tokenizer = UnifiedTokenizer(
            args.img_tokenizer_path,
            txt_tokenizer_type='cogview',
            device=torch.cuda.current_device()
        )
    elif tokenizer_type.startswith('glm'):
        kwargs = {"add_block_symbols": True, "add_task_mask": args.task_mask,
                    "add_decoder_mask": args.block_mask_prob > 0.0}
        if tokenizer_type == "glm_GPT2BPETokenizer":
            from .glm import GPT2BPETokenizer
            get_tokenizer.tokenizer = GPT2BPETokenizer(args.tokenizer_model_type, **kwargs)
        elif tokenizer_type == "glm_ChineseSPTokenizer":
            from .glm import ChineseSPTokenizer
            get_tokenizer.tokenizer = ChineseSPTokenizer(args.tokenizer_model_type, **kwargs)
        elif tokenizer_type == "glm_BertWordPieceTokenizer":
            from .glm import BertWordPieceTokenizer
            get_tokenizer.tokenizer = BertWordPieceTokenizer(args.tokenizer_model_type, **kwargs)
    elif tokenizer_type == 'icetk':
        from icetk import icetk
        get_tokenizer.tokenizer = icetk
    elif tokenizer_type == 'icetk-glm-130B':
        from .icetk_glm_130B import _IceTokenizer
        get_tokenizer.tokenizer = _IceTokenizer()
    # elif tokenizer_type.startswith('hf'):
    #     from .hf_tokenizer import HFT5Tokenizer
    #     if tokenizer_type == "hf_T5Tokenizer":
    #         get_tokenizer.tokenizer = HFT5Tokenizer(args.tokenizer_model_type, cache_dir=args.cache_dir)
    else:
        print_rank0('Try to load tokenizer from Huggingface transformers...')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from transformers import AutoTokenizer
        try:
            get_tokenizer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, trust_remote_code=True)
        except OSError as e:
            print_rank0(f'Cannot find {tokenizer_type} from Huggingface or sat. Creating a fake tokenizer...')
            assert args.vocab_size > 0
            get_tokenizer.tokenizer = FakeTokenizer(args.vocab_size)
            return get_tokenizer.tokenizer
    print_rank0(f'> Set tokenizer as a {tokenizer_type} tokenizer! Now you can get_tokenizer() everywhere.')
    return get_tokenizer.tokenizer


class FakeTokenizer(object):
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def __len__(self):
        return self.num_tokens
