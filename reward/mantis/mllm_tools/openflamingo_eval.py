"""pip install open-flamingo[eval]
"""
import torch
from mantis.models.openflamingo import create_model_and_transforms, OpenFlamingoProcessor
from mantis.models.conversation import conv_openflamingo
from typing import List
from mantis.mllm_tools.mllm_utils import merge_images, load_images
from huggingface_hub import hf_hub_download
import torch

class OpenFlamingo():
    support_multi_image = True
    xatten_map = {
        "anas-awadalla/mpt-1b-redpajama-200b": 1,
        "anas-awadalla/mpt-1b-redpajama-200b-dolly": 1,
        "togethercomputer/RedPajama-INCITE-Base-3B-v1": 2,
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1": 2,
        "anas-awadalla/mpt-7b": 4,
    }
    xatten_map = {
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b": 1,
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct": 1,
        "openflamingo/OpenFlamingo-4B-vitl-rpj3b": 2,
        "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": 2,
        "openflamingo/OpenFlamingo-9B-vitl-mpt7b": 4,
    }
    lang_encoder_map = {
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b": "anas-awadalla/mpt-1b-redpajama-200b",
        "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct": "anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "openflamingo/OpenFlamingo-4B-vitl-rpj3b": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "openflamingo/OpenFlamingo-9B-vitl-mpt7b": "mosaicml/mpt-7b",
    }
    def __init__(self, model_id="openflamingo/OpenFlamingo-9B-vitl-mpt7b", input_type="pretrained") -> None:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=self.lang_encoder_map.get(model_id, "mosaicml/mpt-7b"),
            tokenizer_path=self.lang_encoder_map.get(model_id, "mosaicml/mpt-7b"),
            cross_attn_every_n_layers=self.xatten_map.get(model_id, 4),
        )
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.paddding_side = "left"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        try:
            # original pretrained openflamingo model
            checkpoint_path = hf_hub_download(model_id, "checkpoint.pt")
        except:
            # fine-tuned model on conversational data
            checkpoint_path = hf_hub_download(model_id, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        
        assert input_type in ["pretrained", "chat"], "input_type should be either pretrained or chat"
        self.input_type = input_type
        self.image_token = "<image>"
        self.processor = OpenFlamingoProcessor(self.image_processor, self.tokenizer)
        
    
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
        if self.support_multi_image:
            if self.input_type == "pretrained":
                
                # add one shot for multi choice question eval
                one_shot_inputs = [
                    {
                        "type": "image",
                        "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                    },
                    {
                        "type": "text",
                        "content": "What is in the image?\n (A) a tiger\n (B) a zebra\n (C) a cat\n (D) a dog \n Answer: B"
                    }
                ]
                inputs = one_shot_inputs + inputs
                image_links = []
                text_prompts = []
                _image_links = []
                for x in inputs:
                    if x["type"] == "image":
                        _image_links.append(x["content"])
                    elif x["type"] == "text":
                        text_prompts.append(x["content"])
                        image_links.append(_image_links)
                        _image_links = []
                    else:
                        raise NotImplementedError
                if len(_image_links) > 0:
                    image_links.append(_image_links)
                inputs = self.prepare_prompt(image_links, text_prompts)
                return self.get_parsed_output(inputs)
            elif self.input_type == "chat":
                text_prompt = ""
                for i, message in enumerate(inputs):
                    if message["type"] == "text":
                        text_prompt += message["content"]
                    elif message["type"] == "image":
                        text_prompt += f"{self.image_token} \n"
                image_links = [message["content"] for message in inputs if message["type"] == "image"]
                images = load_images(image_links)
                conv = conv_openflamingo.copy()
                conv.append_message(conv.roles[0], text_prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()
                inputs = self.processor(prompt, images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                inputs["vision_x"] = inputs["vision_x"].to(self.model.dtype)
                output_ids = self.model.generate(
                    vision_x=inputs["vision_x"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    num_beams=3,
                )
                generated_text = self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return generated_text.strip(' \n')
        else:
            raise NotImplementedError
    
    def prepare_prompt(self, image_links:List[List[str]], text_prompt: List[str]):
        if type(image_links) == str:
            image_links = [[image_links]]
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]
            
        assert len(image_links) == len(text_prompt), "Number of images and text prompts should be the same"
        merged_images = [load_images(_image_links) for _image_links in image_links]
        prompt = ""
        for i in range(len(text_prompt)):
            if merged_images[i]:
                prompt += "<image>"*len(merged_images[i]) + text_prompt[i]
            else:
                prompt += " " + text_prompt[i]
        prompt = prompt.strip(' \n')
        all_images = [image for images in merged_images for image in images if image is not None]
        vision_x = [self.image_processor(image).unsqueeze(0) for image in all_images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.model.dtype)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        return {"vision_x": vision_x, "input_ids": input_ids}

    def get_parsed_output(self, inputs):
        vision_x = inputs["vision_x"].to(self.device)
        input_ids = {k: v.to(self.device) for k, v in inputs["input_ids"].items()}
        generated_text = self.model.generate(
            vision_x=vision_x,
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_new_tokens=16,
            num_beams=3,
        )
        decoded_text = self.tokenizer.decode(generated_text[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded_text.strip(' \n')
    
    
if __name__ == "__main__":
    model = OpenFlamingo("Mantis-VL/Mantis-9b-openflamingo_2048", input_type="chat")
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
    Answer: A zebra.
    ### 1 shot
    A cat wearing a bow tie.
    ### 2 shot
    A rabbit.
    """
    