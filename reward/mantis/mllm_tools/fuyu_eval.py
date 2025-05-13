"""need latest transformers from source
pip install transformers>=4.35.2
"""
import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from io import BytesIO
from mantis.mllm_tools.mllm_utils import merge_images

class Fuyu():
    support_multi_image = False
    def __init__(self, model_id:str="adept/fuyu-8b") -> None:
        """
        Args:
            model_id (str): Fuyu model name, e.g. "adept/fuyu-8b"
        """
        self.model_id = model_id
        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.model = FuyuForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
    
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
            raise NotImplementedError
        else:
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            inputs = self.prepare_prompt(image_links, text_prompt)
            return self.get_parsed_output(inputs)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.model.device)
        return inputs
    
    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.pad_token_id)
        input_len = inputs.input_ids.shape[1]
        generation_text = self.processor.batch_decode(generation_output[:, input_len:], skip_special_tokens=True)
        return generation_text[0].strip(" \n")
    
if __name__ == "__main__":
    model = Fuyu()
    # 0 shot
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
    Output: 
    ### 0 shot
    A zebra is in the image.
    ### 1 shot
    A zebra.
    ### 2 shot
    A zebra and a cat.
    """
    