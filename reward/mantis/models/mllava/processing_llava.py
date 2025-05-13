# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Llava.
"""

import os
import json
from typing import List, Optional, Union, Dict

# from ...feature_extraction_utils import BatchFeature
# from ...image_utils import ImageInput
# from ...processing_utils import ProcessorMixin
# from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# from ...utils import TensorType

from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.processing_utils import transformers_module
from transformers.utils.hub import is_remote_url, download_url, cached_file, is_offline_mode
from transformers.utils import IMAGE_PROCESSOR_NAME

from PIL import Image
import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)

class MLlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("CLIPImageProcessor", "SiglipImageProcessor")
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.image_token_index = None
        
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
                    logger.warning(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
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
                        logger.warning(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
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
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        add_image_ids: bool = True,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
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
        if not self.image_token_index:
            self.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        if add_image_ids:
            text, images = self.preprocess_interleaved_images_and_text(text, images)
        
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )
        # text_inputs: 
        # 1. input_ids: [batch_size, sequence_length], e.g. [1, 6]
        # 2. attention_mask: [batch_size, sequence_length], e.g. [1, 6]
        
        # check the number of image token ids, and truncated the number of images if needed
        
        if images is not None:
            input_ids = text_inputs["input_ids"]
            num_image_tokens = torch.sum(input_ids == self.image_token_index, dim=-1)
            for i, num_image_token in enumerate(num_image_tokens):
                if num_image_token < len(images[i]):
                    images[i] = images[i][:num_image_token]
                    print(f"{len(images[i]) - num_image_token} ({len(images[i])} in total) image tokens in the text are truncated due to the max sequence length; removing the extra images.")
            # flatten images
            images = [image for images_per_text in images for image in images_per_text]
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"] # [batch_size, num_channels, height, width], e.g. [1, 3, 336, 336]
        else:
            pixel_values = None
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

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
            if k == "pixel_values":
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            else:
                results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
        return results

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
            
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
        else:
            processor_file = IMAGE_PROCESSOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=True,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {IMAGE_PROCESSOR_NAME} file"
                )
        
        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            image_processor_dict = {}

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            )
            
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                if attribute_name == "tokenizer":
                    classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                    use_fast = kwargs.get("use_fast", True)
                    if use_fast and classes[1] is not None:
                        attribute_class = classes[1]
                    else:
                        attribute_class = classes[0]
                elif attribute_name == "image_processor":
                    image_processor_type = image_processor_dict.get("image_processor_type", None)
                    if image_processor_type is not None:
                        assert image_processor_type in class_name, f"Invalid image processor type: {image_processor_type}"
                        attribute_class = getattr(transformers_module, image_processor_type)
                    else:
                        attribute_class = getattr(transformers_module, class_name[0])
                else:
                    raise ValueError(f"Invalid attribute name: {attribute_name}")
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args
        