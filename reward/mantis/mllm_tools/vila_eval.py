"""
conda create -n vila python=3.10
conda activate vila

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/

# then install mantis for eval, in the root directory of the repo
pip install -e ".[eval]"
"""
import os
import torch
from typing import List
from mantis.mllm_tools.mllm_utils import load_images
import re
import torch

try:

    from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                                IMAGE_TOKEN_INDEX)
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                process_images, tokenizer_image_token)
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except:
    raise ImportError("Please install see mllm_tools/vila_eval.py for running requirements. Due to the vila project's bad compatibilities, you need to install a additional environment for running this code.")

class VILA():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="Efficient-Large-Model/Llama-3-VILA1.5-8b", model_base:str=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, model_base)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "llama-3" in model_name.lower():
            conv_mode = "llama_3"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.conv_template = conv_templates[conv_mode]

        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        """
        image_links = [x["content"] for x in inputs if x["type"] == "image"]
        if self.support_multi_image:
            text_prompt = ""
            for i, message in enumerate(inputs):
                if message["type"] == "text":
                    text_prompt += message["content"]
                elif message["type"] == "image":
                    text_prompt += f"{IMAGE_PLACEHOLDER} "
            images = load_images(image_links)
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in text_prompt:
                if self.model.config.mm_use_im_start_end:
                    text_prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, text_prompt)
                else:
                    text_prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text_prompt)
            else:
                if DEFAULT_IMAGE_TOKEN not in text_prompt:
                    print("no <image> tag found in input. Automatically append one at the beginning of text.")
                    # do not repeatively append the prompt.
                    if self.model.config.mm_use_im_start_end:
                        text_prompt = (image_token_se + "\n") * len(images) + text_prompt
                    else:
                        text_prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + text_prompt
                        
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[
                        images_tensor,
                    ],
                    do_sample=False,
                    temperature=0,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            return outputs
                    
        else:
            raise NotImplementedError
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    
if __name__ == "__main__":
    model = VILA()
    # 0 shot
    zero_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 1 shot
    one_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 2 shot
    two_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A black cat."
        },
        {
            "type": "image",
            "content": "https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    print("### 0 shot")
    print(model(zero_shot_exs))
    print("### 1 shot")
    print(model(one_shot_exs))
    print("### 2 shot")
    print(model(two_shot_exs))
    """
    Output: a tiger and a zebra
    ### 0 shot
    The image features a zebra grazing on grass in a field.
    ### 1 shot
    A black cat.
    ### 2 shot
    A rabbit.
    """
    