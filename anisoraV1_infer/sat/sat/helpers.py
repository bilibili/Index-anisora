# -*- encoding: utf-8 -*-
'''
@File    :   helpers.py
@Time    :   2023/04/10 16:54:36
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import argparse
import textwrap

def print_parser(parser, help_width=32):
    argument_list = []

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if '--help' in action.option_strings:
            continue

        arg_name = ', '.join([opt.lstrip('-') for opt in action.option_strings])
        arg_help = action.help or ''
        arg_type = action.type.__name__ if action.type else 'str'
        arg_default = str(action.default) if action.default is not None else 'None'

        argument_list.append((arg_name, arg_help, arg_type, arg_default))

    max_name_len = max([len(arg[0]) for arg in argument_list])

    print("-" * (max_name_len + 56))
    print(f"{'Argument'.ljust(max_name_len)}  Help" + " "*(help_width-4) + f"{'Type'.ljust(8)}    Default")
    print("-" * (max_name_len + 56))

    wrapper = textwrap.TextWrapper(width=help_width)

    for arg_name, arg_help, arg_type, arg_default in argument_list:
        name_str = arg_name.ljust(max_name_len)
        type_str = arg_type.ljust(8)

        wrapped_help = wrapper.wrap(arg_help)
        if not wrapped_help:
            wrapped_help = ['']

        for i, line in enumerate(wrapped_help):
            if i == 0:
                print(f"{name_str}  {line.ljust(help_width)}  {type_str}  {arg_default}")
            else:
                print(f"{''.ljust(max_name_len)}  {line.ljust(help_width)}")
        print()

def print_aligned_string_list(str_list, column_spacing=2):
    # 获取字符串列表中的最长字符串长度
    max_length = max(len(s) for s in str_list)

    # 计算终端宽度以便我们知道多少列可以显示
    try:
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        terminal_width = 80
    print("-" * (terminal_width-5))

    # 计算每行可容纳的列数
    columns_per_row = (terminal_width + column_spacing) // (max_length + column_spacing)

    # 计算需要多少行来显示所有字符串
    rows_required = (len(str_list) + columns_per_row - 1) // columns_per_row

    # 按行打印字符串，确保对齐
    for row in range(rows_required):
        line = ""
        for col in range(columns_per_row):
            index = row + col * rows_required
            if index < len(str_list):
                line += str_list[index].ljust(max_length + column_spacing)
        print(line.strip())
    print("-" * (terminal_width-5))

def list_avail_models():
    from .model import official
    model_list = []
    for name in dir(official):
        # if is a class
        if isinstance(getattr(official, name), type):
            model_list.append(name)
    print('Available model definitions (example: "from sat import GLMModel"):')
    print_aligned_string_list(model_list)
    return model_list

def list_avail_pretrained():
    from sat.resources.urls import MODEL_URLS
    # iterate over all pretrained models into a list
    model_list = []
    for model_name, model_url in MODEL_URLS.items():
        model_list.append(model_name)
    print('Available pretrained models (example: sat.AutoModel.from_pretrained("roberta-base")):')
    print_aligned_string_list(model_list)
    return model_list

import logging
import torch

def configure_logging():
    logger = logging.getLogger("sat")
    logger.setLevel(os.environ.get("SAT_LOGLEVEL", "INFO"))
    if os.environ.get("LOGLEVEL", None) is not None:
        logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # stream handler
    sh = logging.StreamHandler()
    logger.setLevel(os.environ.get("SAT_LOGLEVEL", "INFO"))
    if os.environ.get("LOGLEVEL", None) is not None:
        logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = configure_logging()

def configure_file_logging(log_path="tmp.txt"):
    logger = logging.getLogger(log_path)
    logger.setLevel(os.environ.get("SAT_LOGLEVEL", "INFO"))
    if os.environ.get("LOGLEVEL", None) is not None:
        logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler = logging.FileHandler("/tmp/" + log_path, encoding='UTF-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

file_logger = {"param.txt":configure_file_logging("param.txt"), "tmp.txt":configure_file_logging("tmp.txt")}

def print_rank0(msg, level=logging.INFO, flush=True, log_file=None):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f"[RANK {torch.distributed.get_rank()}] {msg}"
        if torch.distributed.get_rank() == 0:
            if None != log_file and log_file in file_logger.keys():
                file_logger[log_file].log(level=level, msg=msg)

            logger.log(level=level, msg=msg)
            if flush:
                logger.handlers[0].flush()
    else:
        if None != log_file and log_file in file_logger.keys():
                file_logger[log_file].log(level=level, msg=msg)
        logger.log(level=level, msg=msg)

def print_all(msg, level=logging.INFO, flush=True, log_file=None):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f"[RANK {torch.distributed.get_rank()}] {msg}"

    if None != log_file and log_file in file_logger.keys():
        file_logger[log_file].log(level=level, msg=msg)
    logger.log(level=level, msg=msg)
    if flush:
        logger.handlers[0].flush()

def debug_param(name, param):
    print_all(f"param: {name}, min: {param.min()}, max: {param.max()}, mean: {param.mean()}, std: {param.std()}, scale: {param.abs().mean()}, first5: {param.flatten()[:5]}, last5: {param.flatten()[-5:]}")
    from deepspeed.utils import safe_get_full_grad, safe_get_full_optimizer_state
    g =  safe_get_full_grad(param)
    if g is not None:
        print_all(f"grad: {name}, min: {g.min()}, max: {g.max()}, mean: {g.mean()}, std: {g.std()}, scale: {g.abs().mean()}, first5: {g.flatten()[:5]}, last5: {param.flatten()[-5:]}")
    s = safe_get_full_optimizer_state(param, 'exp_avg')
    if s is not None:
        print_all(f"state: {name}, min: {s.min()}, max: {s.max()}, mean: {s.mean()}, std: {s.std()}, scale: {s.abs().mean()}, first5: {s.flatten()[:5]}")

def get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
    # At this point, the socket is closed, and the port is released
    return port

def check_if_zero3(args):
    return hasattr(args, 'deepspeed_config') and (args.deepspeed_config is not None) and (args.deepspeed_config.get('zero_optimization',{}).get('stage', 0) >= 3)