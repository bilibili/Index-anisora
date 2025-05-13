"""pip install accelerate transformers>=4.35.2
BLIP_FLANT5 tends to otuput shorter text, like "a tiger and a zebra". Try to design the prompt with shorter answer.
"""
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import List
import torch
from typing import List
from io import BytesIO
from mantis.mllm_tools.mllm_utils import merge_images

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class INSTRUCTBLIP_FLANT5():
    support_multi_image = False
    def __init__(self, model_id:str="Salesforce/instructblip-flan-t5-xxl") -> None:
        """
        Args:
            model_id (str): BLIP_FLANT5 model name, e.g. "Salesforce/blip2-flan-t5-xxl"
        """
        self.model_id = model_id
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    
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
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, 
            do_sample=True,
            # num_beams=5,
            max_new_tokens=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].strip(" \n")
    

if __name__ == "__main__":
    model = INSTRUCTBLIP_FLANT5("Salesforce/instructblip-flan-t5-xxl")
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
    In the image, a zebra can be seen grazing in a lush, green field. The zebra is a single animal with long, sleek, and slender legs, making it look graceful and majestic.
    ### 1 shot
    A zebra.
    ### 2 shot
    A rabbit.
    """
    