import requests
import torch
import regex as re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BatchFeature
from typing import List
from mantis.mllm_tools.mllm_utils import load_images, merge_images

class Kosmos2():
    support_multi_image = False
    def __init__(self, model_id:str="microsoft/kosmos-2-patch14-224") -> None:
        """
        Args:
            model_id (str): Kosmos2 model name, e.g. "microsoft/kosmos-2-patch14-224"
        """
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    def process_interleaved_example(self, prompt, images, placeholder="<i>", num_image_tokens=64, add_special_tokens=True, add_eos_token=False, return_tensors=None):
        processor = self.processor

        first_image_token_id = processor.tokenizer.unk_token_id + 1

        image_input_ids = [processor.tokenizer.convert_tokens_to_ids(processor.boi_token)] + list(range(first_image_token_id, num_image_tokens + first_image_token_id)) + [processor.tokenizer.convert_tokens_to_ids(processor.eoi_token)]
        image_attention_mask = [1] * len(image_input_ids)
        # `-2`: not including `boi` and `eoi`
        image_embeds_position_mask = [0] + [1] * (len(image_input_ids) - 2) + [0]

        import re
        components = re.split(rf"({placeholder})", prompt)

        outputs = {"input_ids": [], "attention_mask": [], "image_embeds_position_mask": []}
        for component in components:
            if component != "<i>":
                # add text tokens: no special tokens -> add them at the end
                encoded = processor(text=component, add_special_tokens=False)
                for key in ["input_ids", "attention_mask"]:
                    outputs[key].extend(encoded[key])
                outputs["image_embeds_position_mask"].extend([0] * len(encoded["input_ids"]))
            else:
                # add tokens to indicate image placeholder
                outputs["input_ids"].extend(image_input_ids)
                outputs["attention_mask"].extend(image_attention_mask)
                outputs["image_embeds_position_mask"].extend(image_embeds_position_mask)

        if add_special_tokens:
            outputs["input_ids"] = [processor.tokenizer.bos_token_id] + outputs["input_ids"] + ([processor.tokenizer.eos_token_id] if add_eos_token else [])
            outputs["attention_mask"] = [1] + outputs["attention_mask"] + ([1] if add_eos_token else [])
            outputs["image_embeds_position_mask"] = [0] + outputs["image_embeds_position_mask"] + ([0] if add_eos_token  else [])

        outputs["pixel_values"] = processor.image_processor(images).pixel_values

        for k in ["input_ids", "attention_mask", "image_embeds_position_mask"]:
            outputs[k] = [outputs[k]]
        outputs = BatchFeature(data=outputs,tensor_type=return_tensors)

        return outputs
        
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
            prompt = ""
            for x in inputs:
                if x["type"] == "image":
                    prompt += "<i>"
                elif x["type"] == "text":
                    prompt += "<grounding> " + x["content"]
                else:
                    raise NotImplementedError
            images = load_images([x["content"] for x in inputs if x["type"] == "image"])
            inputs = self.process_interleaved_example(prompt, images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=256,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            generated_text = re.sub(r"<object>(.*)</object>", "", generated_text)
            generated_text = re.sub(r"</?phrase>", "", generated_text)
            return generated_text.strip(" \n")
        else:
            image_links = [x["content"] for x in inputs if x["type"] == "image"]
            merged_image = merge_images(image_links)
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            text_prompt = "<grounding> Question:" + text_prompt +" Answer:"
            
            inputs = self.processor(text=text_prompt, images=merged_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            new_generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip(" \n")
        
if __name__ == "__main__":
    model = Kosmos2()
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
    The image features a zebra grazing on a lush green field. The zebra is standing on its hind legs, grazing on the grass.
    ### 1 shot
    A black cat with blue eyes.
    ### 2 shot
    A white rabbit.
    """
    