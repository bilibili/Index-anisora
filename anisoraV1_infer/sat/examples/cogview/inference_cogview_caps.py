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

from sat import get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed
from sat.model import BaseModel
from sat.generation.sampling_strategies import BaseStrategy
from sat.generation.autoregressive_sampling import get_masks_and_position_ids_default
from sat.generation.utils import generate_continually


def main(args):
    model, args = BaseModel.from_pretrained('cogview-base', args)
    tokenizer = get_tokenizer(args=args)
    
    # define function for each query
    query_template = '[BASE] [BOI1] [Image]{} [EOI1] [ROI1] {}'
    rank = torch.distributed.get_rank()
    output_file = os.path.join(args.output_path, f"scores_rank_{rank}.txt")
    fout = open(output_file, 'w')

    def process(raw_text0):
        raw_text, *imgs = raw_text0.strip().split('\t')
        print('raw text: ', raw_text)
        
        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            input_list = []
            for i in range(tim * mbz, (tim+1)*mbz):
                text = query_template.format(imgs[i], raw_text)
                seq = tokenizer.parse_query(text, img_size=args.img_size)
                if len(seq) > 1088:
                    raise ValueError('text too long.')
                # txt part
                botext = seq.index(tokenizer['[ROI1]'])
                input_list.append(
                    torch.tensor(seq, device=args.device)
                ) 
            batch_input = torch.stack(input_list)
            # forward
            tokens, attention_mask, position_ids = get_masks_and_position_ids_default(batch_input[0])
            attention_mask = attention_mask.type_as(next(model.parameters()))
            tokens = batch_input # get_masks_and_position_ids only accept bz=1
            logits, *mems = model(tokens, position_ids, attention_mask)
            logits = logits.float()
            logits[..., :tokenizer.img_tokenizer.num_tokens] = -float('Inf')
            log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

            pred = log_probs[:, botext:-1, :] 
            target = tokens[:, botext+1:].unsqueeze(-1) 
            scores = torch.gather(pred, dim=2, index=target).squeeze(-1).sum(dim=-1)
            output_list.append(
                scores
                )
        output_tokens = torch.cat(output_list, dim=0)
        fout.write(raw_text0.strip()+'\n')
        fout.write('\t'.join([str(x) for x in output_tokens.tolist()])+'\n')
    
    generate_continually(process, args.input_source)
    fout.close()

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--img-size', type=int, default=256)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)