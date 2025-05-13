# -*- encoding: utf-8 -*-
'''
@File    :   base_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F
from sat.mpu.initialize import get_model_parallel_world_size, get_model_parallel_src_rank, get_model_parallel_group

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
        # convert to 1D
        # logits = logits.view(logits.size()[1])
        logits = logits.contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # logits[indices_to_remove] = filter_value
        # # going back to 2D
        # logits = logits.view(1, -1).contiguous()

        batch_size, vocab_size = logits.shape[:2]
        for i in range(batch_size):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


class BaseStrategy:
    def __init__(self, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0,  repetition_penalty=1., end_tokens=None):
        self.repetition_penalty = repetition_penalty
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = False
        self.context_length = None

    @property
    def is_done(self) -> bool:
        return self._is_done

    def forward(self, logits, tokens, mems, temperature=None, nan_default_token=None):
        if self.context_length is None:
            self.context_length = tokens.shape[-1]
        if temperature is None:
            temperature = self.temperature
        if torch.isnan(logits).any():
            if nan_default_token is None:
                raise ValueError('nan in logits, set nan_default_token to proceed in BaseStrategy.forward.')
            logits.fill_(-1000)
            logits[..., nan_default_token] = 0
        # apply repetition penalty
        penalty_mat = torch.ones_like(logits).float()
        if tokens.shape[-1]> self.context_length:
            penalty_mat.scatter_(1, 
            tokens[:, self.context_length:], torch.ones_like(tokens[:, self.context_length:]).float() * self.repetition_penalty)
        penalty_mat *= temperature
        logits = logits.float() / penalty_mat

        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits, dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        if get_model_parallel_world_size() > 1:
            torch.distributed.broadcast(pred, get_model_parallel_src_rank(), group=get_model_parallel_group())
        if pred.numel() == 1 and pred.item() in self.end_tokens:
            self._is_done = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[0], 1)), dim=1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = False
        self.context_length = None
        return tokens, mems
