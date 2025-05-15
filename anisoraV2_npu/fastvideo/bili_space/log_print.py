# -*- encoding: utf-8 -*-
import os
import argparse
import textwrap
import logging
import torch
from functools import wraps


def only_rank0(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs) if 0 == torch.distributed.get_rank() else None
    return decorated


def shield_prefix(f):
    @wraps(f)
    def decorated(str):
        invalid_prefix = []
        for i in invalid_prefix:
            if str.startswith(i):
                return None
        return f(str)
    return decorated


@shield_prefix
def report_memory(name: str):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print_rank0(string)


@only_rank0
def print_parser_val(parser, args=None, help_width=32):
    if None == args:
        args = parser.parse_args()

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
        arg_val = args.__getattribute__(action.dest)

        argument_list.append((arg_name, arg_help, arg_type, arg_default, arg_val))

    max_name_len = max([len(arg[0]) for arg in argument_list])
    max_default_len = max([len(arg[3]) for arg in argument_list])

    print("-" * (max_name_len + 56))
    print(f"{'Argument'.ljust(max_name_len)}  Help" + " "*(help_width - 4) \
          + f"  {'Type'.ljust(8)}  {'Default'.ljust(max_default_len)}  Val")
    print("-" * (max_name_len + 56))

    wrapper = textwrap.TextWrapper(width=help_width)

    for arg_name, arg_help, arg_type, arg_default, arg_val in argument_list:
        name_str = arg_name.ljust(max_name_len)
        type_str = arg_type.ljust(8)
        arg_default_str = arg_default.ljust(max_default_len)

        wrapped_help = wrapper.wrap(arg_help)
        if not wrapped_help:
            wrapped_help = ['']

        for i, line in enumerate(wrapped_help):
            if i == 0:
                print(f"{name_str}  {line.ljust(help_width)}  {type_str}  {arg_default_str}  {arg_val}")
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
    print("-" * (terminal_width - 5))
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
    print("-" * (terminal_width - 5))


def configure_logging():
    logger = logging.getLogger("default")
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    # stream handler
    sh = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def configure_file_logging(log_path="tmp.txt"):
    logger = logging.getLogger(log_path)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    # file handler
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler = logging.FileHandler("/tmp/" + log_path, encoding='UTF-8')
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


g_logger = configure_logging()
g_file_logger = {"param.txt":configure_file_logging("param.txt"), "tmp.txt":configure_file_logging("tmp.txt")}


def print_rank0(msg, level=logging.INFO, flush=True, log_file=None):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f"[RANK {torch.distributed.get_rank()}] {msg}"
        if torch.distributed.get_rank() == 0:
            if None != log_file and log_file in g_file_logger.keys():
                g_file_logger[log_file].log(level=level, msg=msg)

            g_logger.log(level=level, msg=msg)
            if flush:
                g_logger.handlers[0].flush()
    else:
        if None != log_file and log_file in g_file_logger.keys():
            g_file_logger[log_file].log(level=level, msg=msg)
        g_logger.log(level=level, msg=msg)


def print_all(msg, level=logging.INFO, flush=True, log_file=None):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f"[RANK {torch.distributed.get_rank()}] {msg}"

    if None != log_file and log_file in g_file_logger.keys():
        g_file_logger[log_file].log(level=level, msg=msg)
    g_logger.log(level=level, msg=msg)
    if flush:
        g_logger.handlers[0].flush()


def get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
    # At this point, the socket is closed, and the port is released
    return port
