import functools

import torch


def empty_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        return func(*args, **kwargs)

    return wrapper
