# -*- encoding: utf-8 -*-
'''
@File    :   inference_cogview.py
@Time    :   2021/10/09 19:41:58
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

# from sat import get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed
from sat import get_args, get_tokenizer
from sat.model import CachedAutoregressiveModel
from sat.generation.sampling_strategies import BaseStrategy
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.utils import timed_name, save_multiple_images, generate_continually
from sat.tokenization.cogview import UnifiedTokenizer

def main(args):

    '''
    2022/06/17
    Modify load_checkpoint to from_pretraind
    '''
    # initialize_distributed(args)
    model, args = CachedAutoregressiveModel.from_pretrained('cogview-base', args)
    tokenizer = get_tokenizer(args=args)

    # define function for each query
    query_template = '[ROI1] {} [BASE] [BOI1] [MASK]*1024' if not args.full_query else '{}'
    invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
    strategy = BaseStrategy(invalid_slices,
                            temperature=args.temperature, top_k=args.top_k)
    
    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        print('raw text: ', raw_text)
        text = query_template.format(raw_text)
        seq = tokenizer.parse_query(text, img_size=args.img_size)
        if len(seq) > 1088:
            raise ValueError('text too long.')
        # calibrate text length
        txt_len = seq.index(tokenizer['[BASE]'])
        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=args.device, dtype=torch.half if args.fp16 else torch.float32)
        log_attention_weights[txt_len+2:, 1:txt_len] = 1.8 if txt_len <= 10 else 1.4 # TODO args
        # generation
        seq = torch.cuda.LongTensor(seq, device=args.device)
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            output_list.append(
                filling_sequence(model, seq.clone(),
                    batch_size=min(args.batch_size, mbz),
                    strategy=strategy,
                    log_attention_weights=log_attention_weights
                    )[0]
                )
        output_tokens = torch.cat(output_list, dim=0)
        # decoding
        imgs, txts = [], []
        for seq in output_tokens:
            decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
            imgs.append(decoded_imgs[-1]) # only the last image (target)
        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id)
            os.makedirs(full_path, exist_ok=True)
            save_multiple_images(imgs, full_path, False)
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.jpg', args.output_path)
            save_multiple_images(imgs, full_path, True)
    
    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--full-query', action='store_true')
    py_parser.add_argument('--img-size', type=int, default=256)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)