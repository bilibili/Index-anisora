# -*- encoding: utf-8 -*-
# @File    :   get_dataset.py
# @Time    :   2021/12/14
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

import os
import datasets
from datasets import load_dataset
from sat.helpers import print_rank0

def parse_huggingface_path(path):
    if path.startswith('hf://'):
        path = path[5:]
    names = path.split('/')
    first_name = names[0]
    second_name = names[1] if len(names) >= 2 and names[1] != '*' else None
    split = names[2] if len(names) >= 3 else 'train'
    return first_name, second_name, split

def load_hf_dataset(path, process_fn, columns=None, cache_dir='~/.cache/huggingface/datasets', offline=False, transformer_name = None, rebuild=False):
    dataset_name, sub_name, split = parse_huggingface_path(path)
    datasets.config.HF_DATASETS_OFFLINE = int(offline)
    if transformer_name:
        dataset_path = cache_dir + '/' + dataset_name + "_" + sub_name + "_" + split + "_" + transformer_name + ".data"
    else:
        dataset_path = None

    if dataset_path and os.path.exists(dataset_path) and not rebuild:
        dataset = datasets.load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_name, sub_name, cache_dir=cache_dir, split=split,
        download_config=datasets.utils.DownloadConfig(max_retries=20)) # TODO
        # dataset = dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True)
        print_rank0(f'> Preprocessing the {dataset_name} by process_fn... Next time will return cached files.\n> Pass "rebuild=True" to load_hf_dataset if change process_fn. Change "transformer_name" for different tokenizers or models.')
        dataset = dataset.map(process_fn, batched=False, load_from_cache_file=True)
        if dataset_path:
            dataset.save_to_disk(dataset_path)
    dataset.set_format(type='torch', columns=columns)
    return dataset
