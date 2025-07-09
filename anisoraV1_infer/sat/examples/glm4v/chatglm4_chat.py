import re
import numpy as np
import torch

from transformers import AutoTokenizer
from typing import Dict, List
from sat.helpers import print_rank0

def glm4_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.boi = "[IMG]"
    tokenizer.eoi = "[/IMG]"
    return tokenizer

def format_conversation(conversations, tokenizer, is_inference=False, is_text_only=False):
    # Note: `loss_mask` here means whether *the prediction* of the token should take loss
    tokens, loss_masks = tokenizer.get_prefix_tokens(), [0, 0]

    def _update(_tokens: List[int], value: int = 1):
        value = int(value)
        tokens.extend(_tokens)
        loss_masks.extend([value] * len(_tokens))

    context_length = len(tokens)
    for idx, conv in enumerate(conversations):
        no_training_tokens = []
        # prompt
        no_training_tokens.append(tokenizer.encode("<|user|>", add_special_tokens=False)[0])
        if not is_text_only and idx == 0:
            no_training_tokens.extend([-100]) # img flag
        no_training_tokens.extend(tokenizer.encode(conv["user"], add_special_tokens=False))
        no_training_tokens.append(tokenizer.encode("<|assistant|>", add_special_tokens=False)[0])
        _update(no_training_tokens, 0)
        # context_length
        if idx == len(conversations) - 1:
            context_length = len(tokens)
        # answer
        if not (is_inference and idx == len(conversations) - 1):
            # update answer
            ans_tokens = tokenizer.encode(conv["assistant"], add_special_tokens=False)
            _update(ans_tokens, 1)
            _update([tokenizer.eos_token_id], 1)

    assert len(tokens) == len(loss_masks), f"length mismatch: {len(tokens)} vs {len(loss_masks)}"
    return tokens, loss_masks, context_length

from functools import partial
def get_masks_and_position_ids(seq, image_logits_mask):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = []
    pid = -1
    for i in range(len(image_logits_mask[0])):
        if image_logits_mask[0][i] == 0 or (i > 0 and image_logits_mask[0][i] != image_logits_mask[0][i - 1]):
            pid += 1
        position_ids.append(pid)
    for i in range(tokens.shape[1]-image_logits_mask.shape[1]):
        pid += 1
        position_ids.append(pid)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids

class chatglm4_text_processor_inference:
    def __init__(self, tokenizer, max_target_length=1024, image_length=257, model=None, no_prompt=False, english=True):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        self.sep = "<|assistant|>"
        self.invalid_slices = []
        self.no_eoi = True

    def __call__(self, query, history=[], **kwargs):
        """
        Args:
            history (list): [(q1, a1), (q2, a2)]
            query (str, optional): questions
        """
        prompt = self.history_to_prompt(history, query)
        input_ids, _, _ = format_conversation(prompt, self.tokenizer, is_inference=True)
        while -100 in input_ids:
            img_idx = input_ids.index(-100)
            input_ids = input_ids[:img_idx] + [0] * (self.image_length + 1) + [-1] + input_ids[img_idx + 1:]

        image_position = []
        while -1 in input_ids:
            img_idx = input_ids.index(-1)
            input_ids[img_idx] = 0
            image_position.append(img_idx)

        image_embed_mask = [0] * len(input_ids)
        vision_expert_mask = [0] * len(input_ids)
        rope_mask = [0] * len(input_ids)
        for idx in image_position:
            image_embed_mask[idx - self.image_length - 1: idx + 1] = [1] * (self.image_length + 2)
            vision_expert_mask[idx - self.image_length - 1: idx] = [1] * (self.image_length + 1)
            rope_mask[idx - self.image_length: idx] = [1] * self.image_length

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        image_embed_mask = torch.tensor(image_embed_mask).unsqueeze(0)
        vision_expert_mask = torch.tensor(vision_expert_mask).unsqueeze(0)
        rope_mask = torch.tensor(rope_mask).unsqueeze(0)
        return {'input_ids': input_ids, 'image_embed_mask': image_embed_mask, 'rope_mask': rope_mask}

    def history_to_prompt(self, history, query):
        ret = []
        for i, (old_query, response) in enumerate(history):
            ret.append({"user": str(old_query).strip(), "assistant": str(response).strip()})
        ret.append({"user": str(query).strip()})
        return ret

    def get_func(self, inputs, **kwargs):
        get_func = partial(get_masks_and_position_ids, image_logits_mask=kwargs['rope_mask'])
        return get_func
