# -*- encoding: utf-8 -*-
'''
@File    :   beam_search_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import torch
import torch.nn.functional as F
from .base_strategy import top_k_logits
from sat.mpu.initialize import get_model_parallel_world_size, get_model_parallel_src_rank, get_model_parallel_group

class BeamSearchStrategy:
    def __init__(self, num_beams, length_penalty=1., 
                temperature=1., top_k=0., top_p=0.0,
                consider_end=True,
                end_tokens=[], invalid_slices=[], no_repeat_ngram_size=0, 
                min_tgt_length=0,
                repetition_penalty=1.,
                prefer_min_length=5,
                prefer_max_length=100,
                stop_n_iter_unchanged=10):
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_tgt_length = min_tgt_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.stop_n_iter_unchanged = stop_n_iter_unchanged
        self.prefer_min_length = prefer_min_length
        self.prefer_max_length = prefer_max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._init_cache()

    def _init_cache(self):
        self.end_beams = [] # list of LongTensors
        self.end_beams_penalized_scores = [] # list of LongTensors
        self.cached_beam_scores = 0 # [batch_size]
        self.cached_beam_ngram_bans = [{} for i in range(self.num_beams)]
        self.is_done = False
        self.end_beams_unchanged = 0
        self.context_length = None
    
    def _add_end_beams(self, score, beam):
        gen_length = len(beam) - self.context_length # we usually care about generated length, instead of total length
        # score = score / ((5. + gen_length) / 6) ** self.length_penalty # Magic number for OpenNMT 
        # ----
        trunc_length = min(self.prefer_max_length, gen_length) + 1
        if gen_length >= self.prefer_min_length:
            adjust_penalty = self.length_penalty
        else:
            t = gen_length / self.prefer_min_length
            min_penalty = min(0.5, self.length_penalty / 2)
            adjust_penalty = t * self.length_penalty + (1-t) * min_penalty
        # print(float(score), trunc_length, adjust_penalty)
        score = float(score) / trunc_length ** adjust_penalty
        # ----
        
        for i in range(len(self.end_beams), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[i-1]:
                break
        self.end_beams.insert(i, beam)
        self.end_beams_penalized_scores.insert(i, score)
        self.end_beams = self.end_beams[:self.num_beams]
        self.end_beams_penalized_scores = self.end_beams_penalized_scores[:self.num_beams]
        # print('add end beam', score, i)
        return (i == 0)

    def forward(self, logits, tokens, mems):
        batch_size, vocab_size = logits.shape
        seq_len = tokens.shape[-1]
        if self.context_length is None:
            self.context_length = seq_len

        logits = logits.float()
        penalty_mat = torch.ones_like(logits)
        if tokens.shape[-1]> self.context_length:
            penalty_mat.scatter_(1, 
            tokens[:, self.context_length:], torch.ones_like(tokens[:, self.context_length:]).float() * self.repetition_penalty)  
        penalty_mat *= self.temperature
        logits = logits.float() / penalty_mat

        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_tgt_length > seq_len:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for i in range(batch_size):
                ngram_prefix = tokens[i, -(self.ngram-1):].tolist() # TODO ngram=1
                for banned_index in self.cached_beam_ngram_bans[i].get(tuple(ngram_prefix), []):
                    logits[i, banned_index] = -65504
        
        # logits = logits / self.temperature
        logits = top_k_logits(logits, self.top_k, self.top_p)

        next_token_scores = F.log_softmax(logits, dim=-1) # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(self.cached_beam_scores, torch.Tensor):
            prev_scores = prev_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores + prev_scores
        
        next_token_scores = next_token_scores.view(batch_size * vocab_size)

        probs = F.softmax(logits.view(batch_size * vocab_size), dim=0)
        next_tokens = torch.multinomial(probs, 
            num_samples=(max(1,len(self.end_tokens))+1) * self.num_beams) # [2*nb]
        if get_model_parallel_world_size() > 1:
            torch.distributed.broadcast(next_tokens, get_model_parallel_src_rank(), group=get_model_parallel_group())
        next_token_scores = next_token_scores[next_tokens]
        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=0)
        next_tokens = next_tokens[_indices]

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode='trunc')
        next_tokens = next_tokens % vocab_size

        # select out end beams or continue beams
        if mems.shape[1] < batch_size:
            mems = mems.expand(-1, batch_size, -1, -1)
        beam_continue = []
        scores_continue = []
        bans_continue = []
        mems_contiue = []
        end_beams_changed = False
        for i in range(len(next_tokens)):
            beam = torch.cat((tokens[next_indices[i]], next_tokens[i:i+1]))
            if int(next_tokens[i]) in self.end_tokens:
                changed = self._add_end_beams(next_token_scores[i], beam)
                end_beams_changed = end_beams_changed or changed
            elif len(beam_continue) < self.num_beams:
                beam_continue.append(beam)
                mems_contiue.append(mems[:, next_indices[i]])
                # update caches
                scores_continue.append(next_token_scores[i])
                if self.ngram > 0:
                    bans = self.cached_beam_ngram_bans[next_indices[i]].copy()
                    ngram_prefix = tuple(tokens[next_indices[i], -(self.ngram-1):].tolist()) # TODO ngram=1
                    bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (next_tokens[i],)
                    bans_continue.append(bans)
            else:
                break
        tokens = torch.stack(beam_continue)
        mems = torch.stack(mems_contiue, dim=1)
        self.cached_beam_scores = torch.tensor(scores_continue, device=logits.device)
        self.cached_beam_ngram_bans = bans_continue

        # check if done, this is not a official solution
        if end_beams_changed:
            self.end_beams_unchanged = 0
        elif len(self.end_beams) > 0:
            self.end_beams_unchanged += 1
            if self.end_beams_unchanged >= self.stop_n_iter_unchanged:
                self.is_done = True

        return tokens, mems

    def finalize(self, tokens, mems):
        if self.consider_end:
            if not self.is_done:
                for i in range(tokens.shape[0]):
                    self._add_end_beams(self.cached_beam_scores[i], tokens[i])
            mems = None
            ret = self.end_beams
        else:
            ret = tokens
        self._init_cache()
        return ret, mems
