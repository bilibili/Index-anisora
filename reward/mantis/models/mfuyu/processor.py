import torch
import numpy as np
from transformers import FuyuProcessor
from transformers.models.fuyu.processing_fuyu import (
    BEGINNING_OF_ANSWER_STRING,
    _transform_coordinates_and_tokenize,
    full_unpacked_stream_to_tensor,
    logger,
    requires_backends,
    FuyuBatchFeature,
)
from transformers import TensorType
from transformers.models.fuyu.image_processing_fuyu import (
    PILImageResampling,
    ChannelDimension,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_list_of_images,
    to_channel_dimension_format,
    to_numpy_array,
    get_image_size,
)
from typing import List, Union, Optional, Dict, Tuple
from PIL import Image
from tqdm import tqdm

BEGINNING_OF_IMAGE = "|IMAGE|"
END_OF_IMAGE = "|ENDOFIMAGE|" # special token
IGNORE_INDEX = -100

def MFuyuImageProcessor_preprocess(
    self,
    images,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: Optional[PILImageResampling] = None,
    do_pad: Optional[bool] = None,
    padding_value: Optional[float] = None,
    padding_mode: Optional[str] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[float] = None,
    image_std: Optional[float] = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    patch_size: Optional[Dict[str, int]] = None,
    data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_tensors: Optional[TensorType] = None,
):
    """

    Utility function to preprocess the images and extract necessary information about original formats.

    Args:
        images (`ImageInput`):
            Images to preprocess. Expects a single image, a list or images or a list of lists of images. Pixel
            values range from 0 to 255, or between 0 and 1 if `do_rescale` is `False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image to `size`.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
        do_pad (`bool`, *optional*, defaults to `self.do_pad`):
            Whether to pad the image to `size`.
        padding_value (`float`, *optional*, defaults to `self.padding_value`):
            The value to pad the image with.
        padding_mode (`str`, *optional*, defaults to `self.padding_mode`):
            The padding mode to use when padding the image.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float`, *optional*, defaults to `self.image_mean`):
            The mean to use when normalizing the image.
        image_std (`float`, *optional*, defaults to `self.image_std`):
            The standard deviation to use when normalizing the image.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            The factor to use when rescaling the image.
        patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        return_tensors (`str` or `TensorType`, *optional*):
            The type of tensors to return. Can be one of:
            - Unset: Return a list of `np.ndarray`.
            - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
            - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
            - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format of the output image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
    """

    do_resize = do_resize if do_resize is not None else self.do_resize
    size = size if size is not None else self.size
    resample = resample if resample is not None else self.resample
    do_pad = do_pad if do_pad is not None else self.do_pad
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else self.do_normalize
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    padding_value = padding_value if padding_value is not None else self.padding_value
    padding_mode = padding_mode if padding_mode is not None else self.padding_mode
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    patch_size = patch_size if patch_size is not None else self.patch_size

    batch_images = make_list_of_list_of_images(images)

    if do_resize and size is None:
        raise ValueError("Size must be specified if do_resize is True.")

    if do_rescale and rescale_factor is None:
        raise ValueError("Rescale factor must be specified if do_rescale is True.")

    if do_normalize and image_mean is None or image_std is None:
        raise ValueError("image_mean and image_std must be specified if do_normalize is True.")

    # All transformations expect numpy arrays.
    batch_images = [[to_numpy_array(image) for image in images] for images in batch_images]

    if is_scaled_image(batch_images[0][0]) and do_rescale:
        logger.warning_once(
            "It looks like you are trying to rescale already rescaled images. If the input"
            " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        )

    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(batch_images[0][0])

    original_image_sizes = [[get_image_size(image, channel_dim=input_data_format) for image in images] for images in batch_images]
    # original_image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]

    if do_resize:
        batch_images = [
            [self.resize(image, size=size, input_data_format=input_data_format) for image in images]
            for images in batch_images
        ]

    # image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]
    # image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
    # image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]
    image_sizes = [[get_image_size(image, channel_dim=input_data_format) for image in images] for images in batch_images]
    image_unpadded_heights = [[image_size[0] for image_size in image_sizes] for image_sizes in image_sizes]
    image_unpadded_widths = [[image_size[1] for image_size in image_sizes] for image_sizes in image_sizes]
    

    # scale_h is the same as scale_w
    # image_scale_factors = [
    #     [resized_size[0] / original_size[0]]
    #     for original_size, resized_size in zip(original_image_sizes, image_sizes)
    # ]
    image_scale_factors = [
        [resized_size[0] / original_size[0] for original_size, resized_size in zip(original_sizes, sizes)] 
        for original_sizes, sizes in zip(original_image_sizes, image_sizes)
    ]

    if do_pad:
        batch_images = [
            [
                self.pad_image(
                    image,
                    size=size,
                    mode=padding_mode,
                    constant_values=padding_value,
                    input_data_format=input_data_format,
                )
                for image in images
            ]
            for images in batch_images
        ]

    if do_rescale:
        batch_images = [
            [self.rescale(image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
            for images in batch_images
        ]

    if do_normalize:
        batch_images = [
            [
                self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]
            for images in batch_images
        ]

    if data_format is not None:
        batch_images = [
            [to_channel_dimension_format(image, data_format, input_data_format) for image in images]
            for images in batch_images
        ]

    data = {
        "images": batch_images,
        "image_unpadded_heights": image_unpadded_heights,
        "image_unpadded_widths": image_unpadded_widths,
        "image_scale_factors": image_scale_factors,
    }
    return FuyuBatchFeature(data=data, tensor_type=return_tensors)


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    scale_factors: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
    add_IMGSEP: bool = False,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    # If not tool use, tranform the coordinates while tokenizing
    if scale_factors is not None:
        transformed_prompt_tokens = []
        for prompt_seq, scale_factor_seq in zip(prompts, scale_factors):
            seq_tokens = []
            for i, prompt in enumerate(prompt_seq):
                if i < len(scale_factor_seq):
                    seq_tokens.append(_transform_coordinates_and_tokenize(prompt, scale_factor_seq[i].item(), tokenizer))
                else:
                    seq_tokens.append(tokenizer(prompt, add_special_tokens=False).input_ids)
            transformed_prompt_tokens.append(seq_tokens)
    else:
        transformed_prompt_tokens = [[tokenizer(prompt, add_special_tokens=False).input_ids for prompt in prompt_seq] for prompt_seq in prompts]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        bos_token = tokenizer.vocab["<s>"]
    else:
        bos_token = tokenizer.vocab["|ENDOFTEXT|"]
    
    if add_IMGSEP:
        img_bos_token = tokenizer.vocab[BEGINNING_OF_IMAGE]
        img_eos_token = tokenizer.vocab[END_OF_IMAGE]
        new_prompts_tokens = []
        for prompt_seq in prompts_tokens:
            new_prompt_seq = []
            for i, x in enumerate(prompt_seq):
                _x = []
                if add_BOS and len(x) > 0:
                    _x.append(bos_token)
                if i == 0:
                    new_prompt_seq.append(_x + x + [img_bos_token])
                elif i == len(prompt_seq) - 1:
                    new_prompt_seq.append([img_eos_token] + _x + x)
                else:
                    new_prompt_seq.append([img_eos_token] + _x + x + [img_bos_token])
            new_prompts_tokens.append(new_prompt_seq)
        prompts_tokens = new_prompts_tokens
    else:
        prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]
        
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # Only add bbox open token to the last subsequence since that is what will be completed
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)
            
    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.

    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len: int = np.max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        logger.warning(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError("Length of subsequence prompt exceeds sequence length.")
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab["|ENDOFTEXT|"]] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    return prompts_tokens_tensor, prompts_length_tensor

def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    all_bi_stream = []

    for batch_index in range(batch_size):
        all_si_stream = []

        # First, construct full token stream (including image placeholder tokens) and loss mask for each subsequence
        # and append to lists. We use lists rather than tensors because each subsequence is variable-sized.
        # TODO Remove this logic in a subsequent release since subsequences are not supported.
        si_stream_len = 0
        for i in range(num_real_text_tokens[batch_index].shape[0]):
            if num_sub_sequences and si_stream_len + num_real_text_tokens[batch_index][i] > num_sub_sequences:
                break
            all_si_stream.append(input_stream[batch_index, i][:num_real_text_tokens[batch_index][i]])
            si_stream_len += num_real_text_tokens[batch_index][i]
            if i < len(image_tokens[batch_index]):
                if num_sub_sequences and si_stream_len + image_tokens[batch_index][i].shape[0] > num_sub_sequences:
                    break
                # add image placeholder tokens and special tokens indicating the beginning and the end of the image
                all_si_stream.append(image_tokens[batch_index][i])
                si_stream_len += image_tokens[batch_index][i].shape[0]
        if len(all_si_stream) == 0:
            print('Warning: the first subsequece exceeds the maximum length of the sequence. Empty sequence is returned. This may because of the big image size or long prompt length.')
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    return all_bi_stream

class MFuyuProcessor(FuyuProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor.__class__.preprocess = MFuyuImageProcessor_preprocess # change the preprocess function to our own
        self.add_BOS = True
        self.add_IMGSEP = True
        self.add_beginning_of_answer_token = False
        self.empty_image = Image.new("RGB", (256, 256), color="black")
        self.tokenizer.add_special_tokens({"additional_special_tokens": [BEGINNING_OF_IMAGE, END_OF_IMAGE]})
        
    
    def get_sample_encoding(
        self,
        prompts,
        scale_factors,
        image_unpadded_heights,
        image_unpadded_widths,
        image_placeholder_id,
        image_newline_id,
        tensor_batch_images,
    ):
        batch_size = len(prompts)
        # image_present = torch.ones(1, 1, 1)
        image_present = torch.ones(batch_size, tensor_batch_images.shape[1], 1)
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )
        # FIXME max_tokens_to_generate is embedded into this processor's call.
        prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_position_embeddings=self.max_position_embeddings,
            add_BOS=self.add_BOS,
            add_beginning_of_answer_token=self.add_beginning_of_answer_token,
            add_IMGSEP=self.add_IMGSEP,
        )
        image_padded_unpacked_tokens = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=prompt_tokens,
            image_tokens=model_image_input["image_input_ids"],
            batch_size=batch_size,
            num_sub_sequences=self.subsequence_length,
        )
        # Construct inputs for image patch indices.
        unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
            num_real_text_tokens=prompts_length,
            input_stream=torch.full_like(prompt_tokens, -1),
            image_tokens=model_image_input["image_patch_indices_per_batch"],
            batch_size=batch_size,
            num_sub_sequences=self.subsequence_length,
        )
        max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[0].shape[0]))

        # Use same packing logic for the image patch indices.
        image_patch_input_indices = full_unpacked_stream_to_tensor(
            all_bi_tokens_to_place=[tokens_to_place],
            full_unpacked_stream=unpacked_image_patch_indices_per_batch,
            fill_value=-1,
            batch_size=batch_size,
            new_seq_len=max_seq_len_batch,
            offset=0,
        )
        # image_patches_tensor = torch.stack([img[0] for img in model_image_input["image_patches"]])
        image_patches_tensor = torch.stack([torch.cat(imgs) for imgs in model_image_input["image_patches"]])
        
        batch_encoding = {
            "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
            "image_patches": image_patches_tensor,
            "image_patches_indices": image_patch_input_indices,
        }
        return batch_encoding
    
    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> "FuyuBatchFeature":
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

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

        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ["torch"])

        # --- Check input validity ---
        if not return_attention_mask:
            raise ValueError("`return_attention_mask=False` is not supported for this model.")
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be None.")
        if text is not None and images is None:
            # logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            text_encoding["image_patches"] = [torch.zeros(
                    [1, 1, 3 * self.image_processor.patch_size["height"] * self.image_processor.patch_size["width"]],
                    dtype=torch.int32)] * len(text_encoding["input_ids"])
            text_encoding["image_patches_indices"] = torch.zeros([len(text_encoding["input_ids"]), 1], dtype=torch.int32) - 1
            return text_encoding

        if text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            if isinstance(images, Image.Image):
                images = [images]
            if isinstance(images, list) and isinstance(images[0], Image.Image):
                images = [images]
            texts = ["<image>"*len(image) for image in images]
            
        if text is not None and images is not None:
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
                        logger.warning("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
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
            
        prompts = [[t for t in text.split("<image>")] for text in texts]
        for i, prompt_seq in enumerate(prompts):
            for j in range(len(prompt_seq) - 1):
                prompt_seq[j] += f"image {j}: "

        
        # --- Use self.tokenizer to get the ids of special tokens to insert into image ids ---
        image_placeholder_id = self.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
        image_newline_id = self.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
        
        # FIXME - We hard code "pt" here because the rest of the processing assumes torch tensors
        self.subsequence_length = max_length
        all_encodings = []
        for prompt, sub_images in zip(prompts, images):
            if not sub_images:
                text_encoding = self.tokenizer(
                    text=prompt,
                    add_special_tokens=add_special_tokens,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    stride=stride,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=False,
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_offsets_mapping=return_offsets_mapping,
                    return_token_type_ids=return_token_type_ids,
                    return_length=return_length,
                    verbose=verbose,
                    return_tensors=return_tensors,
                    **kwargs,
                )
                image_encoding = self.image_processor.preprocess([self.empty_image], return_tensors="pt")
                text_encoding["image_patches"] = torch.zeros(
                    [1, 1, 3 * self.image_processor.patch_size["height"] * self.image_processor.patch_size["width"]],
                    dtype=torch.int32)
                text_encoding["image_patches_indices"] = torch.zeros([1, 1], dtype=torch.int32) - 1
                
                all_encodings.append(text_encoding)
            else:
                # --- Preprocess images using self.image_processor ---
                if not sub_images:
                    image_encoding = self.image_processor.preprocess([self.empty_image], return_tensors="pt")
                else:
                    image_encoding = self.image_processor.preprocess([sub_images], return_tensors="pt")
                batch_images = torch.stack(image_encoding["images"][0], dim=0).unsqueeze(0)
                self.batch_size = len(batch_images)
                image_unpadded_heights = torch.tensor(image_encoding["image_unpadded_heights"])
                image_unpadded_widths = torch.tensor(image_encoding["image_unpadded_widths"])
                scale_factors = image_encoding["image_scale_factors"]
                sample_encoding = self.get_sample_encoding(
                    prompts=[prompt],
                    scale_factors=scale_factors,
                    image_unpadded_heights=image_unpadded_heights,
                    image_unpadded_widths=image_unpadded_widths,
                    image_placeholder_id=image_placeholder_id,
                    image_newline_id=image_newline_id,
                    tensor_batch_images=batch_images,
                )
                all_encodings.append(sample_encoding)
            
        batch_encoding = self._left_pad_inputs_with_attention_mask(
            model_inputs=all_encodings, return_attention_mask=return_attention_mask
        )
        return FuyuBatchFeature(data=batch_encoding)
    
    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool=True, max_length=None):
        max_length_input_ids = max(entry["input_ids"].shape[1] for entry in model_inputs)
        max_length_input_ids = max_length if max_length is not None else max_length_input_ids
        max_length_image_patch_indices = max(entry["image_patches_indices"].shape[1] for entry in model_inputs)

        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}
        if "labels" in model_inputs[0]:
            batched_inputs["labels"] = []

        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == "input_ids":
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    padded_input_ids = torch.cat(
                        [
                            tensor,
                            torch.full((tensor.shape[0], num_padding_tokens), self.pad_token_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_input_ids)

                    attention_mask = torch.cat(
                        [torch.ones_like(tensor), torch.zeros(tensor.shape[0], num_padding_tokens, dtype=torch.long)],
                        dim=1,
                    )
                    batched_inputs["attention_mask"].append(attention_mask)

                elif key == "image_patches":
                    # For image_patches, we don't pad but just append them to the list.
                    batched_inputs[key].append(tensor)
                
                elif key == "labels":
                    num_padding_tokens = max_length_input_ids - tensor.shape[1]
                    padded_labels = torch.cat(
                        [
                            tensor,
                            torch.full((tensor.shape[0], num_padding_tokens), IGNORE_INDEX, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_labels)

                else:  # for image_patches_indices
                    num_padding_indices = max_length_image_patch_indices - tensor.shape[1]
                    padded_indices = torch.cat(
                        [
                            tensor,
                            torch.full(
                                (tensor.shape[0], num_padding_indices), self.dummy_image_index, dtype=torch.long
                            ),
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_indices)
        batched_keys = ["input_ids", "image_patches_indices"] + (["labels"] if "labels" in batched_inputs else [])
        if return_attention_mask:
            batched_keys.append("attention_mask")
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        return batched_inputs