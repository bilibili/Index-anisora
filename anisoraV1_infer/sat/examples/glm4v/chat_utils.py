# -*- encoding: utf-8 -*-
import os
import sys
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, stream_filling_sequence, BaseStrategy, get_masks_and_position_ids_default
from sat.mpu import get_model_parallel_rank

def process_image(text, text_processor, img_processor, image=None):
    '''Process image in text.
    Args:
        text: str, text.
        image: Optional, image path / url / PIL image.
    '''
    image_position = text.rfind(text_processor.tokenizer.boi) + 5
    if image_position < 5:
        return text, image_position, (None, None, None)
    # extract path from [IMG][/IMG] using re
    pattern = (text_processor.tokenizer.boi + r"(.*?)" + text_processor.tokenizer.eoi).replace('[', r'\[').replace(']', r'\]')
    image_path = re.findall(pattern, text)
    image_path = image_path[-1] if image_path[-1] else None
    if image is None:
        assert image_path is not None, "image and image_path cannot be both None."
        text = text.replace(image_path, "")
        image_path = image_path.strip()
        # url
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        # local path
        else:
            image = Image.open(image_path)
    if image is not None and isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
        image = img_processor(pil_img) if img_processor is not None else {}
        # image = image.unsqueeze(0)
        ret = (image, pil_img)
    else:
        ret = image
    return text, image_position, ret


def chat(image_path, model, text_processor, img_processor,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 1024, top_p=0.7, top_k=30, temperature=0.95, repetition_penalty=1.2, english=True,
        invalid_slices=[], force_pil_image=None, args=None, sample_strategy=None
        ):
    is_image_mode = image_path or (type(image) is not tuple and image is not None) or (type(image) is tuple and image != (None, None)) or force_pil_image is not None
    if not history:
        history = []
    if is_image_mode and not force_pil_image:
        prompt = "{}{}{}".format(text_processor.tokenizer.boi, image_path if image_path else "", text_processor.tokenizer.eoi)
    else:
        prompt = ""
    if force_pil_image is not None:
        image_position = 0
        torch_image = img_processor(force_pil_image) if img_processor is not None else {}
        pil_img = force_pil_image
    else:
        prompt, image_position, (torch_image, pil_img) = process_image(prompt, text_processor, img_processor, image=image)
    prompt = query
    if torch_image is not None:
        assert type(torch_image) is dict
        if type(torch_image) is dict:
            for k in torch_image:
                if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                    torch_image[k] = torch_image[k].to(torch.bfloat16 if args.bf16 else torch.float16)
                if type(torch_image[k]) is torch.Tensor:
                    torch_image[k] = torch_image[k].to(next(model.parameters()).device)
        else:
            torch_image = torch_image.to(torch.bfloat16 if args.bf16 else torch.float16).to(next(model.parameters()).device)
        
    inputs_dic = text_processor(prompt, history=history, is_text_only=not is_image_mode)
    for k in inputs_dic:
        if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
            inputs_dic[k] = inputs_dic[k].to(torch.bfloat16 if args.bf16 else torch.float16)
        if type(inputs_dic[k]) is torch.Tensor:
            inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
    inputs = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
    origin = inputs
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty) if sample_strategy is None else sample_strategy
    get_func = text_processor.get_func(inputs, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default
    if image_position < 5:
        inputs_dic.pop('input_ids')
        inputs = inputs_dic
    else:
        inputs = {'vision_'+k:v for k,v in torch_image.items()}
        inputs_dic.pop('input_ids')
        inputs = {**inputs, **inputs_dic}
        # breakpoint()
    
    if getattr(args, "stream_chat", False):
        filling_stream = stream_filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            **inputs
        )
        if get_model_parallel_rank() == 0:
            if english:
                print("Model: ", end='')
            else:
                print("模型：", end='')
        offset = len(text_processor.tokenizer.decode(origin))
        for tokens, mems in filling_stream:
            torch.cuda.empty_cache()
            tmp_response = text_processor.tokenizer.decode(tokens[0])
            if tmp_response[-1] != "�":
                if get_model_parallel_rank() == 0:
                    print(tmp_response[offset:], end='')
                offset = len(tmp_response)
        if get_model_parallel_rank() == 0:
            print()
        output = strategy.finalize(tokens, mems)[0]

        response = text_processor.tokenizer.decode(output[0])

    else:
        # offset = len(text_processor.tokenizer.decode(origin))
        output = filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            **inputs
        )[0] # drop memory
        
        # ---------------
        # port from inference_glm.py, more general than chat mode
        # clip -1s and fill back generated things into seq
        if type(output) is not list:
            output_list = output.tolist()
        else:
            output_list = output

        response = text_processor.tokenizer.decode(output_list[0])
    response = response.split(text_processor.sep)[-1].strip()
    history = history + [(query, response)]
    return response, history, (torch_image, pil_img)
