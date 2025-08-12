import math
import copy
import os
import warnings
import re
import sys
from transformers.utils import logging
logger = logging.get_logger(__name__)

import torch
import torch.nn as nn
from transformers import GenerationMixin
from sat import AutoModel
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.generation.logits_process import LogitsProcessor
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores

from transformers import AutoConfig

class ChatModel(nn.Module, GenerationMixin):
    def __init__(self, args, model=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained('THUDM/chatglm2-6b', trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if model is None:
            self.model, self.args = AutoModel.from_pretrained("chatglm2-6b", args)
        else:
            self.model, self.args = model, args
        self.device = self.model.parameters().__next__().device
        self.main_input_name = 'input_ids'
    
    @classmethod
    def from_pretrained(cls, name, args=None, base_cls=None, *, home_path=None, url=None, prefix='', **kwargs):
        if base_cls is None:
            model, args = AutoModel.from_pretrained(name, args, home_path=home_path, url=url, prefix=prefix, **kwargs)
        else:
            model, args = base_cls.from_pretrained(name, args, home_path=home_path, url=url, prefix=prefix, **kwargs)
        return cls(args, model), args
    
    def can_generate(self):
        return True

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        if past is None:
            past = past_key_values
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "return_last_logit": True
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            return_last_logit: Optional[bool] = False,
            **kw_args
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        lm_logits = outputs[0]
        if return_last_logit:
            lm_logits = lm_logits[:, -1:]
        past_key_values = [x['past_key_values'] for x in outputs[1:]]

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=past_key_values
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response
    
    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history
    
    @torch.no_grad()
    def batch_generate(self, tokenizer, queries, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, **kwargs):
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}
        inputs = tokenizer(queries, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        outputs = self.generate(**inputs, **gen_kwargs)
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts
