"""pip install transformers>=4.35.2 transformers_stream_generator torchvision tiktoken chardet matplotlib
""" 
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
class QwenVL():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_id:str="Qwen/Qwen-VL-Chat") -> None:
        """
        Args:
            model_id (str): Qwen model name, e.g. "Qwen/Qwen-VL-Chat"
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, bf16=True).eval()
    
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
        true_image_links = []
        for i, image_link in enumerate(image_links):
            if isinstance(image_link, str):
                true_image_links.append(image_link)
            elif isinstance(image_link, Image.Image):
                image_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                image_file.close()
                image_link.save(image_file.name)
                self.merged_image_files.append(image_file.name)
                true_image_links.append(image_file.name)
            else:
                raise NotImplementedError
        image_links = true_image_links
        input_list = []
        for i, image_link in enumerate(image_links):
            input_list.append({'image': image_link})
        input_list.append({'text': text_prompt})
        query = self.tokenizer.from_list_format(input_list)
        return query
    

    def get_parsed_output(self, query):
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def __del__(self):
        for image_file in self.merged_image_files:
            os.remove(image_file)
    
if __name__ == "__main__":
    model = QwenVL()
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
    Output: a tiger and a zebra
    ### 0 shot
    This image is a photo of a zebra grazing on grass. The zebra is standing on a green grass field and its head is slightly lowered, focusing on eating grass. The stripes on the zebra's body are alternating black and white.
    ### 1 shot
    A black cat with blue eyes wearing a small orange bow tie.
    ### 2 shot
    A white rabbit.
    """
    