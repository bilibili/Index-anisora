import os
import copy
import functools
import json
import logging
import math
import multiprocessing
import random
import time
import hashlib
import base64
import openai
import numpy as np
import tiktoken
import tqdm
from collections import namedtuple
from typing import Optional, Sequence
from pathlib import Path
from openai import OpenAI, AzureOpenAI
from typing import Optional, Sequence, Union
from mimetypes import guess_type
from io import BytesIO
from PIL import Image

__all__ = ["openai_completions"]
tiktoken.model.MODEL_TO_ENCODING['ChatGPT'] = tiktoken.model.MODEL_TO_ENCODING['gpt-3.5-turbo']
# API specific
DEFAULT_OPENAI_API_BASE = openai.base_url
OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", None))
if isinstance(OPENAI_API_KEYS, str):
    OPENAI_API_KEYS = OPENAI_API_KEYS.split(",")
OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")
OPENAI_MAX_CONCURRENCY = int(os.environ.get("OPENAI_MAX_CONCURRENCY", 5))
CLIENT_CLASS = AzureOpenAI if "azure" == openai.api_type else OpenAI

def get_cache_dir():
    import os
    from pathlib import Path
    cache_dir = os.path.join(str(Path.home()), ".easy-openai")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

cache_dir = Path(get_cache_dir()) / "cache"
cache_dir.mkdir(exist_ok=True)
cache_base_path = None
cache_base = None

print("Default cache dir:", cache_dir)

def get_prompt_uids(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def openai_completions(
    prompts: Union[Sequence[str], Sequence[Sequence[dict]]],
    model_name: str,
    tokens_to_favor: Optional[Sequence[str]] = None,
    tokens_to_avoid: Optional[Sequence[str]] = None,
    is_skip_multi_tokens_to_avoid: bool = True,
    is_strip: bool = True,
    num_procs: Optional[int] = OPENAI_MAX_CONCURRENCY,
    batch_size: Optional[int] = None,
    use_cache: bool = True,
    return_json: bool = False,
    **decoding_kwargs,
) -> dict[str, list]:
    r"""Get openai completions for the given prompts. Allows additional parameters such as tokens to avoid and
    tokens to favor.

    Parameters
    ----------
    prompts : list of str or list of messages
        Prompts to get completions for.
        

    model_name : str
        Name of the model to use for decoding.

    tokens_to_favor : list of str, optional
        Substrings to favor in the completions. We will add a positive bias to the logits of the tokens constituting
        the substrings.

    tokens_to_avoid : list of str, optional
        Substrings to avoid in the completions. We will add a large negative bias to the logits of the tokens
        constituting the substrings.

    is_skip_multi_tokens_to_avoid : bool, optional
        Whether to skip substrings from tokens_to_avoid that are constituted by more than one token => avoid undesired
        side effects on other tokens.

    is_strip : bool, optional
        Whether to strip trailing and leading spaces from the prompts.

    use_cache : bool, optional
        Whether to use cache to save the query results in case of multiple queries.
        
    return_json : bool
        Whether to ask chatGPT to return json formatted string or not

    decoding_kwargs :
        Additional kwargs to pass to `openai.Completion` or `openai.ChatCompletion`.

    Example
    -------
    >>> prompts = ["Respond with one digit: 1+1=", "Respond with one digit: 2+2="]
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_avoid=["2"," 2"])['completions']
    ['\n\nAnswer: \n\nTwo (or, alternatively, the number "two" or the numeral "two").', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003", tokens_to_favor=["2"])['completions']
    ['2\n\n2', '\n\n4']
    >>> openai_completions(prompts, model_name="text-davinci-003",
    ... tokens_to_avoid=["2 a long sentence that is not a token"])['completions']
    ['\n\n2', '\n\n4']
    >>> chat_prompt = ["<|im_start|>user\n1+1=<|im_end|>", "<|im_start|>user\nRespond with one digit: 2+2=<|im_end|>"]
    >>> openai_completions(chat_prompt, "gpt-3.5-turbo", tokens_to_avoid=["2"," 2"])['completions']
    ['As an AI language model, I can confirm that 1+1 equals  02 in octal numeral system, 10 in decimal numeral
    system, and  02 in hexadecimal numeral system.', '4']
    """
    # add cache support for query
    num_procs = num_procs or OPENAI_MAX_CONCURRENCY
    
    assert isinstance(prompts, list) and (
        isinstance(prompts[0], str) or (isinstance(prompts[0], list) and isinstance(prompts[0][0], dict))
    ), "prompts must be a list of str or a list of list of dict"
    if isinstance(prompts[0], str):
        if "<|im_start|>" not in prompts[0] and _requires_chatml(model_name):
            # requires chatml
            prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
            prompts = [_chatml_to_prompt(prompt) for prompt in prompts]
    else:
        prompts = [_chatml_to_prompt(prompt) for prompt in prompts]

    if use_cache:
        global cache_base
        global cache_base_path
        cache_base_path = cache_dir / f"{model_name}.jsonl"
        if cache_base is None:
            if not cache_base_path.exists():
                cache_base = {}
                logging.warning(
                    f"Cache file {cache_base_path} does not exist. Creating new cache.")
            else:
                with open(cache_base_path, "r") as f:
                    cache_base = [json.loads(line) for line in f.readlines()]
                cache_base = {item['uid']: item for item in cache_base}
                logging.warning(f"Loaded cache base from {cache_base_path}.")

    n_examples = len(prompts)
    if n_examples == 0:
        logging.warning("No samples to annotate.")
        return []
    else:
        logging.warning(
            f"Using `openai_completions` on {n_examples} prompts using {model_name}.")

    if tokens_to_avoid or tokens_to_favor:
        tokenizer = tiktoken.encoding_for_model(model_name)

        logit_bias = decoding_kwargs.get("logit_bias", {})
        if tokens_to_avoid is not None:
            for t in tokens_to_avoid:
                curr_tokens = tokenizer.encode(t)
                if len(curr_tokens) != 1 and is_skip_multi_tokens_to_avoid:
                    logging.warning(
                        f"'{t}' has more than one token, skipping because `is_skip_multi_tokens_to_avoid`.")
                    continue
                for tok_id in curr_tokens:
                    logit_bias[tok_id] = -100  # avoids certain tokens

        if tokens_to_favor is not None:
            for t in tokens_to_favor:
                curr_tokens = tokenizer.encode(t)
                for tok_id in curr_tokens:
                    # increase log prob of tokens to match
                    logit_bias[tok_id] = 7

        decoding_kwargs["logit_bias"] = logit_bias

    if is_strip:
        prompts = [p.strip() for p in prompts]

    is_chat = decoding_kwargs.get(
        "requires_chatml", _requires_chatml(model_name))
    if is_chat:
        # prompts = [_prompt_to_chatml(prompt) for prompt in prompts]
        num_procs = num_procs or 4
        batch_size = batch_size or 1

        if batch_size > 1:
            logging.warning(
                "batch_size > 1 is not supported yet for chat models. Setting to 1")
            batch_size = 1

    else:
        num_procs = num_procs or 1
        batch_size = batch_size or 10

    # logging.warning(f"Kwargs to completion: {decoding_kwargs}"
    n_batches = int(math.ceil(n_examples / batch_size))

    prompt_batches = [
        prompts[batch_id * batch_size: (batch_id + 1) * batch_size] for batch_id in range(n_batches)]

    if "azure" == openai.api_type:
        # Azure API uses engine instead of model
        kwargs = dict(n=1, model=model_name, is_chat=is_chat,
                      use_cache=use_cache, **decoding_kwargs)
    else:
        # OpenAI API uses model instead of engine
        kwargs = dict(n=1, model=model_name, is_chat=is_chat,
                      use_cache=use_cache, **decoding_kwargs)
    
    if use_cache:
        kwargs["cache_base"] = cache_base
        kwargs["cache_base_path"] = cache_base_path
    # logging.warning(f"Kwargs to completion: {kwargs}")

    with Timer() as t:
        if num_procs == 1 or n_examples == 1:
            completions = [
                _openai_completion_helper(prompt_batch, **kwargs)
                for prompt_batch in tqdm.tqdm(prompt_batches, desc="prompt_batches", total=len(prompt_batches), disable=len(prompt_batches) == 1)
            ]
        else:
            with multiprocessing.Pool(num_procs) as p:
                partial_completion_helper = functools.partial(
                    _openai_completion_helper, **kwargs)
                completions = list(
                    tqdm.tqdm(
                        p.imap(partial_completion_helper, prompt_batches),
                        desc="prompt_batches",
                        total=len(prompt_batches),
                        disable=len(prompt_batches) == 1,
                    )
                )
    if n_examples > 1:
        logging.warning(f"Completed {n_examples} examples in {t}.")

    # flatten the list and select only the text
    completions_text = [completion['content']
                        for completion_batch in completions for completion in completion_batch]

    price = [
        completion["total_tokens"] * _get_price_per_token(model_name)
        if completion["total_tokens"] is not None else 0
        for completion_batch in completions
        for completion in completion_batch
    ]
    avg_time = [t.duration / n_examples] * len(completions_text)

    return dict(completions=completions_text, price_per_example=price, time_per_example=avg_time)


def _openai_completion_helper(
    prompt_batch: Sequence[str],
    is_chat: bool,
    sleep_time: int = 2,
    openai_organization_ids: Optional[Sequence[str]] = OPENAI_ORGANIZATION_IDS,
    openai_api_keys: Optional[Sequence[str]] = OPENAI_API_KEYS,
    openai_api_base: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    top_p: Optional[float] = 1.0,
    temperature: Optional[float] = 0.7,
    use_cache: bool = True,
    cache_base: Optional[dict] = None,
    cache_base_path: Optional[str] = None,
    return_json: bool = False,
    **kwargs,
):

    client_kwargs = dict()

    # randomly select orgs
    if openai_organization_ids is not None:
        client_kwargs["organization"] = random.choice(openai_organization_ids)

    openai_api_keys = openai_api_keys or OPENAI_API_KEYS

    if openai_api_keys is not None:
        client_kwargs["api_key"] = random.choice(openai_api_keys)

    # set api base
    client_kwargs["base_url"] = base_url = openai_api_base if openai_api_base is not None else DEFAULT_OPENAI_API_BASE

    client = CLIENT_CLASS(**client_kwargs)
    
    # copy shared_kwargs to avoid modifying it
    kwargs.update(dict(max_tokens=max_tokens,
                  top_p=top_p, temperature=temperature))
    curr_kwargs = copy.deepcopy(kwargs)

    if use_cache:
        prompt_uids = [get_prompt_uids(prompt) for prompt in prompt_batch]
        cache_completions = [cache_base[prompt_uid]['completion']
                             if prompt_uid in cache_base else None for prompt_uid in prompt_uids]
        to_query_prompt_batch = [prompt for prompt, cache_completion in zip(
            prompt_batch, cache_completions) if cache_completion is None]
    else:
        to_query_prompt_batch = prompt_batch
    if is_chat:
        to_query_prompt_batch = [_prompt_to_chatml(
            prompt) for prompt in to_query_prompt_batch]

    # now_cand = ""
    # retry_times = 0
    if len(to_query_prompt_batch) != 0:
        while True:
            try:
                if is_chat:
                    # print(curr_kwargs)
                    if return_json:
                        completion_batch = client.chat.completions.create(messages=to_query_prompt_batch[0], response_format={ "type": "json_object" }, **curr_kwargs)
                    else:
                        completion_batch = client.chat.completions.create(messages=to_query_prompt_batch[0], **curr_kwargs)

                    choices = completion_batch.choices
                    for choice in choices:
                        assert choice.message.role == "assistant"

                else:
                    if return_json:
                        completion_batch = client.completions.create(prompt=to_query_prompt_batch, response_format={ "type": "json_object" }, **curr_kwargs)
                    else:
                        completion_batch = client.completions.create(prompt=to_query_prompt_batch, **curr_kwargs)
                    choices = completion_batch.choices

                batch_avg_tokens = completion_batch.usage.total_tokens / len(prompt_batch)
                break
            except openai.OpenAIError as e:
                if "Please reduce your prompt" in str(e):
                    kwargs["max_tokens"] = int(kwargs["max_tokens"] * 0.8)
                    logging.warning(
                        f"Reducing target length to {kwargs['max_tokens']}, Retrying...")
                    if kwargs["max_tokens"] == 0:
                        logging.exception(
                            "Prompt is already longer than max context length. Error:")
                        raise e
                else:
                    if "rate limit" in str(e).lower():
                        pass
                        # print(e)
                    elif "ResponsibleAIPolicyViolation" in str(e):
                        logging.error("Responsible AI Policy Violation, return empty completions.")
                        logging.error("Details: ", e)
                        Choice = namedtuple("Choice", ["message"])
                        Message = namedtuple("Message", ["content"])
                        choices = [Choice(message=Message(content=""))] * len(to_query_prompt_batch)
                        batch_avg_tokens = 0
                        break
                    else:
                        logging.warning(
                            f"Unknown error {e}. \n It's likely a rate limit so we are retrying...")
                    if openai_organization_ids is not None and len(openai_organization_ids) > 1:
                        client_kwargs["organization"] = organization = random.choice(
                            [o for o in openai_organization_ids if o != openai.organization]
                        )
                        client = CLIENT_CLASS(**client_kwargs)
                        logging.info(f"Switching OAI organization.")
                    if openai_api_keys is not None and len(openai_api_keys) > 1:
                        client_kwargs["api_key"] = random.choice([o for o in openai_api_keys if o != openai.api_key])
                        client = CLIENT_CLASS(**client_kwargs)
                        logging.info(f"Switching OAI API key.")
                    logging.info(f"Sleeping {sleep_time} before retrying to call openai API...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if use_cache:
        responses = []
        to_cache_items = []
        to_query_idx = 0
        for i in range(len(prompt_batch)):
            prompt_uid = prompt_uids[i]
            if cache_completions[i] is None:
                cache_base[prompt_uid] = dict(uid=prompt_uid,
                                              prompt=prompt_batch[i], completion=choices[to_query_idx].message.content,
                                              top_p=top_p, temperature=temperature, max_tokens=max_tokens, total_tokens=batch_avg_tokens)
                to_cache_items.append(cache_base[prompt_uid])
                responses.append(dict(
                    content=choices[to_query_idx].message.content, total_tokens=batch_avg_tokens))
                to_query_idx += 1
            else:
                responses.append(
                    dict(content=cache_completions[i], total_tokens=None))
        assert to_query_idx == len(to_query_prompt_batch)
        # save cache items
        with open(cache_base_path, "a+") as f:
            for item in to_cache_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        responses = [dict(content=choice.message.content, total_tokens=batch_avg_tokens) for choice in choices]
    return responses


def _requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    # TODO: this should ideally be an OpenAI function... Maybe it already exists?
    return "turbo" in model.lower() or "gpt-4" in model.lower() or "chatgpt" in model.lower() or "gpt-4" in model.lower()


def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> _prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        try:
            content = eval(newline_splitted[1].split(end_token, 1)[0].strip())
        except SyntaxError:
            content = newline_splitted[1].split(end_token, 1)[0].strip()
        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message

# Function to encode a local image into data URL 
def local_image_to_data_url(image:Union[str, Image.Image, Path]) -> str:
    if isinstance(image, Path) and image.exists() or isinstance(image, str) and os.path.exists(image):
        image_path = image
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/jpeg'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
        
    elif isinstance(image, Image.Image):
        dummy_path = f"temp.{image.format}"
        mime_type, _ = guess_type(dummy_path)
        if mime_type is None:
            mime_type = 'image/jpeg'
            image_format = "JPEG"
        else:
            image_format = mime_type.split("/")[-1].upper()
        # encode the image
        with BytesIO() as output:
            image.save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, str) and (image.startswith("http") or image.startswith("data:")):
        return image
    else:
        raise ValueError("Image must be a path to a local image, an image object, or a URL.")

def _chatml_to_prompt(message: Sequence[dict], start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a ChatML message to a text prompt

    Examples
    --------
    >>> message = [
    ...     {'content': 'You are a helpful assistant.', 'role': 'system'},
    ...     {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
    ...     {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
    ...     {'content': 'Orange.', 'role': 'user'}
    ...     {'content': [
                {"type": "text", "text": "What's in the image?"},
                {"type": "image_url", "image_url": "https://example.com/image.jpg"}
            ]}
    ... ]
    >>> _chatml_to_prompt(message)
    '<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho\'s there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>'
    """
    prompt = ""
    for m in message:
        role = m["role"]
        name = m.get("name", None)
        if name is not None:
            role += f" name={name}"
        if isinstance(m["content"], list):
            for content in m["content"]:
                if content["type"] == "image_url":
                    if isinstance(content["image_url"], str):
                        content['image_url'] = {
                            "url": local_image_to_data_url(content["image_url"])
                        }
                    elif isinstance(content["image_url"], dict):
                        if isinstance(content["image_url"]["url"], str) or isinstance(content["image_url"]["url"], Image.Image):
                            content["image_url"]["url"] = local_image_to_data_url(content["image_url"]["url"])
                        else:
                            raise ValueError("image_url must be a string or a Image object")
                    else:
                        raise ValueError("image_url must be a string or a dictionary")
                elif content["type"] == "image":
                    if isinstance(content["image"], str):
                        content['type'] = 'image_url'
                        content['image_url'] = {"url": local_image_to_data_url(content["image"])}
                        del content["image"]
                    elif isinstance(content["image"], dict):
                        assert isinstance(content["image"]["url"], str), "image must be a string"
                        content['type'] = 'image_url'
                        content['image_url'] = {}
                        content["image_url"]["url"] = local_image_to_data_url(content["image"]["url"])
                        del content["image"]
                elif content["type"] == "text":
                    pass
                else:
                    raise ValueError(f"Unknown content type {content['type']} in message.")
        prompt += f"<|im_start|>{role}\n{m['content']}\n<|im_end|>\n"
    return prompt


def _string_to_dict(to_convert):
    r"""Converts a string with equal signs to dictionary. E.g.
    >>> _string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def _get_price_per_token(model):
    """Returns the price per token for a given model"""
    if "gpt-4" in model:
        return (
            0.03 / 1000
        )  # that's not completely true because decoding is 0.06 but close enough given that most is context
    elif "gpt-3.5-turbo" in model.lower() or 'chatgpt' in model.lower() or "gpt-35-turbo" in model.lower():
        return 0.002 / 1000
    elif "text-davinci-003" in model:
        return 0.02 / 1000
    else:
        logging.warning(
            f"Unknown model {model} for computing price per token.")
        return np.nan



class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"

# # Example usage with ChatGPT Azure Model
# prompts = ["Respond with one digit: 1+1=", "Respond with one digit: 2+2="]
# chatmls = [[{"role":"system","content":"You are an AI assistant that helps people find information."},
#             {"role":"user","content": prompt}] for prompt in prompts]
# chatml_prompts = [_chatml_to_prompt(chatml) for chatml in chatmls]
# print(chatml_prompts)
# openai_completions(chatml_prompts, model_name="ChatGPT")['completions']
