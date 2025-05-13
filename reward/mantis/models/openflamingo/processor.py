import torch

from typing import List, Union, Optional
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from transformers.utils import TensorType

DEFAULT_IMAGE_TOKEN = "<image>"

def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0).unsqueeze(1)
    return image
    
    
class OpenFlamingoProcessor():
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
    
    def __call__(
        self,
        text: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]], None],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ):
        if images is not None:
            if isinstance(text, str):
                text = [text]
                if isinstance(images, Image.Image):
                    images = [[images]]
                elif isinstance(images, list) and isinstance(images[0], Image.Image):
                    images = [images]
                elif isinstance(images, list) and isinstance(images[0], list) and isinstance(images[0][0], Image.Image):
                    if len(images) != 1:
                        raise ValueError("Assuming multiple images per text. but got multiple image groups, but only one text. number of image groups: {}".format(len(images)))
                    pass
                else:
                    raise ValueError("Invalid images input when text is a string.")
            elif isinstance(text, list):
                assert all(isinstance(t, str) for t in text)
                if isinstance(images, Image.Image):
                    if len(text) == 1:
                        images = [[images]]
                    else:
                        raise ValueError("Multiple Images provided but only one text.")
                elif isinstance(images, list) and isinstance(images[0], Image.Image):
                    if len(text) != len(images):
                        raise ValueError("Assuming one image per text. but got different number of images and text. number of images: {}, number of text: {}".format(len(images), len(text)))
                    images = [[img] for img in images]
                elif isinstance(images, list) and isinstance(images[0], list) and isinstance(images[0][0], Image.Image):
                    if len(text) != len(images):
                        raise ValueError("Assuming multiple image per text. but got different number of image groups and text. number of image groups: {}, number of text: {}".format(len(images), len(text)))
                    pass
                else:
                    raise ValueError("Invalid images input when text is a list of strings.")
            else:
                raise ValueError("Invalid text input.")
            
            pixel_values = torch.stack([preprocess_image(img_group, self.image_processor) for img_group in images], dim=0)
        else:
            pixel_values = None
            
        # assert number of <image> tokens is equal to number of images
        for i in range(len(text)):
            if text[i].count(DEFAULT_IMAGE_TOKEN) < (len(images[i]) if images is not None else 0):
                print("Number of <image> tokens in text is less than number of images. automatically correcting.")
                # replace the first <image> token with (len(images[i]) - text[i].count(DEFAULT_IMAGE_TOKEN)) <image> tokens
                text[i] = text[i].replace(DEFAULT_IMAGE_TOKEN, (len(images[i]) - text[i].count(DEFAULT_IMAGE_TOKEN)) * DEFAULT_IMAGE_TOKEN, 1)
            elif text[i].count(DEFAULT_IMAGE_TOKEN) > (len(images[i]) if images is not None else 0):
                print("Number of <image> tokens in text is more than number of images. automatically correcting.")
                # remove extra <image> tokens at the 
                text[i] = text[i].replace(DEFAULT_IMAGE_TOKEN, "", text[i].count(DEFAULT_IMAGE_TOKEN) - (len(images[i]) if images is not None else 0))
            
            assert text[i].count(DEFAULT_IMAGE_TOKEN) == (len(images[i]) if images is not None else 0)
                
        text_tensor = self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
            
        return BatchFeature(data={
            "input_ids": text_tensor["input_ids"],
            "attention_mask": text_tensor["attention_mask"],
            "vision_x": pixel_values,
        })
        
        
        
        
                