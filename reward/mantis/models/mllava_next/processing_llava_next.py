# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for LLaVa-NeXT.
"""

import torch
from typing import List, Optional, Union, Dict

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging
from PIL import Image

logger = logging.get_logger(__name__)


class MLlavaNextProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-NeXT processor which wraps a LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LlavaNextImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    def preprocess_interleaved_images_and_text(
        self,
        text,
        images=None,
    ):
        """
        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                text can contain <image> tokens as the placeholder for the image(s) to be inserted.
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`, `List[List[PIL.Image.Image]]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
                the number of the images should match the number of <image> tokens in the text.
        
        """
        assert text is not None, "text cannot be None."
            
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            if isinstance(images, list) and isinstance(images[0], Image.Image):
                if isinstance(text, str):
                    images = [images]
                elif isinstance(text, list):
                    if len(text) != len(images):
                        raise ValueError("Invalid input text. Number of texts does not match number of images.")
                    images = [[image] for image in images]
            if isinstance(text, str):
                num_images = len(images[0])    
                num_image_tokens = text.count("<image>")
                if num_image_tokens < num_images:
                    # prepend empty image tokens to text
                    if "USER:" in text:
                        text = text.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "Human:" in text:
                        text = text.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "HUMAN:" in text:
                        text = text.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                    else:
                        text = "<image>" * (num_images - num_image_tokens) + text
                    # logger.warning("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                elif num_image_tokens > num_images:
                    text = text.split("<image>")
                    for i, t in enumerate(text):
                        if i < num_images:
                            text[i] = t + "<image>"
                    text = "".join(text)
                    logger.warning("Number of <image> tokens exceeds number of images. Automatically removing extra tokens at the end of the text.")
                    # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                for i, t in enumerate(text):
                    num_image_tokens = t.count("<image>")
                    num_images = len(images[i])
                    if num_image_tokens < num_images:
                        # prepend empty image tokens to text
                        if "USER:" in t:
                            t = t.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "Human:" in t:
                            t = t.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "HUMAN:" in t:
                            t = t.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                        else:
                            t = "<image>" * (num_images - num_image_tokens) + t
                        # logger.warning("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                    elif num_image_tokens > num_images:
                        t = t.split("<image>")
                        for j, s in enumerate(t):
                            if j < num_images:
                                t[j] = s + "<image>"
                        t = "".join(t)
                        logger.warning("Number of <image> tokens exceeds number of images. Automatically removing extra tokens at the end of the text.")
                        # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                    text[i] = t
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
            assert all([t.count("<image>") == len(images_per_text) for t, images_per_text in zip(texts, images)]), "Number of <image> tokens in text does not match number of images."
            # add image denotation in text before each <image> as "(image {i}: <image>)"
            for i, t in enumerate(texts):
                for j in range(len(images[i])):
                    t = t.replace("<image>", f"(image {j+1}: <Image><IMAGE></Image>)", 1)
                t = t.replace("<IMAGE>", "<image>")
                texts[i] = t
            
            # flatten images
            images = [image for images_per_text in images for image in images_per_text]
        else:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
        
        return texts, images
    
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        texts, images = self.preprocess_interleaved_images_and_text(text, images)

        if images is not None:
            image_inputs = [self.image_processor(image, return_tensors=return_tensors) for image in images]
            image_inputs = {
                "pixel_values": [x["pixel_values"][0] for x in image_inputs],
                "image_sizes": torch.stack([x["image_sizes"][0] for x in image_inputs]),
            }
            # image_inputs = self.image_processor(images, return_tensors=return_tensors)
        else:
            image_inputs = {}
        text_inputs = self.tokenizer(
            texts, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict]):
        results = {}
        assert len(model_inputs) == 1, "This method only supports a single input, but get {} inputs".format(len(model_inputs))
        for k in model_inputs[0].keys():
            if model_inputs[0][k] is not None:
                if isinstance(model_inputs[0][k], torch.Tensor):
                    results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
                elif isinstance(model_inputs[0][k], list):
                    for i, inputs in enumerate(model_inputs):
                        if i == 0:
                            results[k] = inputs[k]
                        else:
                            results[k].extend(inputs[k])
            else:
                results[k] = None
        return results
        
