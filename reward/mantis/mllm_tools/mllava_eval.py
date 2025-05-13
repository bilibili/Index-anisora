"""pip install transformers>=4.35.2
"""
import os
import torch
from mantis.mllm_tools.mllm_utils import merge_images, load_images
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava
from typing import List
from transformers.utils import is_flash_attn_2_available

class MLlava():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="Mantis-VL/mllava_nlvr2_4096") -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        attn_implementation = None
        print(f"Using {attn_implementation} for attention implementation")
        if "llava_next" in model_path:
            from mantis.models.mllava_next import MLlavaNextProcessor, LlavaNextForConditionalGeneration
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation).eval()
            self.processor = MLlavaNextProcessor.from_pretrained(model_path)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation).eval()
            self.processor = MLlavaProcessor.from_pretrained(model_path)

        
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
        generation_kwargs = {
            "max_new_tokens": 4096,
            "num_beams": 1,
            "do_sample": False,
        }
        if self.support_multi_image:
            text_prompt = ""
            for i, message in enumerate(inputs):
                if message["type"] == "text":
                    text_prompt += message["content"]
                elif message["type"] == "image":
                    text_prompt += f"<image> "
            images = load_images(image_links)
            return chat_mllava(text=text_prompt, images=images, model=self.model, processor=self.processor, **generation_kwargs)[0]
        else:
            merged_image = merge_images(image_links)
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            text_prompt = self.conv_template.format(text_prompt)
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Generate
            generate_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated_text
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    
if __name__ == "__main__":
    model = MLlava()
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
    A zebra.
    ### 2 shot
    A black cat.
    """
    