"""need latest transformers from source
pip install transformers>=4.35.2
"""
import torch
from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor, chat_mfuyu
from typing import List
from mantis.mllm_tools.mllm_utils import load_images 

class MFuyu():
    support_multi_image = True
    def __init__(self, model_id:str="Mantis-VL/mfuyu_v2_8192_720p-5500") -> None:
        """
        Args:
            model_id (str): Fuyu model name, e.g. "adept/fuyu-8b"
        """
        self.model_id = model_id
        self.processor = MFuyuProcessor.from_pretrained(model_id)
        self.model = MFuyuForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        if '480p' in model_id:
            self.processor.image_processor.size = {"height": 480, "width": 660}
        elif '720p' in model_id:
            self.processor.image_processor.size = {"height": 720, "width": 1290}
        elif '1080p' in model_id:
            self.processor.image_processor.size = {"height": 1080, "width": 1920}
        else:
            self.processor.image_processor.size = {"height": 1080, "width": 1920}
            
    
    def __call__(self, inputs: List[dict]) -> str:
        """
        Args:
            Only for sinlge turn!
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
            text_prompt = "<image>".join([x["content"] for x in inputs if x["type"] == "text"])
            inputs = self.prepare_prompt(image_links, text_prompt)
            return self.get_parsed_output(inputs)
        else:
            raise NotImplementedError

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        images = load_images(image_links)
        inputs = {
            "text": text_prompt,
            "images": images,
        }
        return inputs
    
    def get_parsed_output(self, inputs):
        generation_kwargs = {
            "max_new_tokens": 4096,
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        generated_text, _ = chat_mfuyu(inputs["text"], inputs["images"], self.model, self.processor, **generation_kwargs)
        return generated_text
    
if __name__ == "__main__":
    model = MFuyu()
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
    