# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/10/09 17:18:26
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import time
import stat
from datetime import datetime
from torchvision.utils import save_image
import torch.distributed as dist
from sat.mpu import get_data_parallel_world_size, get_data_parallel_rank, get_model_parallel_rank


def timed_name(prefix, suffix=None, path=None):
    return os.path.join(
        path, 
        f"{prefix}-{datetime.now().strftime('%m-%d-%H-%M-%S')}{suffix}"
    )

def save_multiple_images(imgs, path, debug=True):
    # imgs: list of tensor images
    if debug:
        imgs = torch.cat(imgs, dim=0)
        print("\nSave to: ", path, flush=True)
        save_image(imgs, path, normalize=True)
    else:
        print("\nSave to: ", path, flush=True)
        for i in range(len(imgs)):
            save_image(imgs[i], os.path.join(path, f'{i}.jpg'), normalize=True)
            os.chmod(os.path.join(path,f'{i}.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)
        save_image(torch.cat(imgs, dim=0), os.path.join(path,f'concat.jpg'), normalize=True)
        os.chmod(os.path.join(path,f'concat.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)

def generate_continually(func, input_source='interactive'):
    if input_source == 'interactive':
        while True:
            raw_text, is_stop = "", False
            if torch.distributed.get_rank() == 0:
                raw_text = input("\nPlease Input Query (stop to exit) >>> ")
                raw_text = raw_text.strip()
                if not raw_text:
                    print('Query should not be empty!')
                    continue
                if raw_text == "stop":
                    is_stop = True
                torch.distributed.broadcast_object_list([raw_text, is_stop])
            else:
                info = [raw_text, is_stop]
                torch.distributed.broadcast_object_list(info)
                raw_text, is_stop = info
            if is_stop:
                return
            try:
                start_time = time.time()
                func(raw_text)
                if torch.distributed.get_rank() == 0:
                    print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue
    else:
        with open(input_source, 'r', encoding="utf-8") as fin:
            inputs = fin.readlines()
        for line_no, raw_text in enumerate(inputs):
            if line_no % get_data_parallel_world_size() != get_data_parallel_rank():
                continue
            rk = dist.get_rank()
            if get_model_parallel_rank() == 0:
                print(f'Working on No. {line_no} on model group {rk}... ')
            raw_text = raw_text.strip()
            if len(raw_text) == 0:
                continue
            start_time = time.time()
            func(raw_text)
            if get_model_parallel_rank() == 0:
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
