"""pip install timm sentencepiece
"""
import os
import torch
import time
import torch.nn as nn
from typing import List, Union, Optional, Dict
from transformers.image_utils import load_image
from mantis.easy_openai import openai_completions





class GPT4V():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="gpt-4v") -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        self.model_path = model_path

        
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
            images = [load_image(image_link) for image_link in image_links]
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                }
            ]
            messages += [
                {
                    "role": "user",
                    "content": []
                }
            ]
            for message in inputs:
                if message["type"] == "image":
                    messages[-1]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": message["content"]
                            }
                        }
                    )
                elif message["type"] == "text":
                    messages[-1]["content"].append(
                        {
                            "type": "text",
                            "text": message["content"]
                        }
                    )
                else:
                    raise NotImplementedError
                

            results = openai_completions(
                [messages],
                model_name=self.model_path,
                temperature=0.0,
                top_p=1.0,
            )
            response = results['completions'][0]

            return response
        else:
            raise NotImplementedError
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)


if __name__ == "__main__":
    model = GPT4V()
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
    # difference
    difference_exs = [
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
        },
    ]
    # print("### 0 shot")
    # print(model(zero_shot_exs))
    # print("### 1 shot")
    # print(model(one_shot_exs))
    # print("### 2 shot")
    # print(model(two_shot_exs))
    print("### difference")
    print(model(difference_exs))
    """
    Output: a tiger and a zebra
    ### 0 shot
    The image features a zebra grazing on grass in a field.
    ### 1 shot
    A zebra.
    ### 2 shot
    A black cat.
    """
    