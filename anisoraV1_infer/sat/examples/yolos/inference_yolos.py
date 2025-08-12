import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets_.coco import build, CocoDetection
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
import skimage
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from skimage import io
import argparse
import datasets_.transforms as T
import copy
import glob
import re
torch.set_grad_enabled(False)

from infer_util import *

os.environ["SAT_HOME"] = "/data/qingsong/pretrain"

swiss_args = argparse.Namespace(
    image_size=[512, 864],
    pre_len=1,
    post_len=100,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=False,
    checkpoint_num_layers=1,
    sandwich_ln=False,
    post_ln=False,
    model_parallel_size=1,
    world_size=1,
    rank=0,
    old_checkpoint=None,
    layernorm_order='pre',
    mode='inference',
    fp16=False,
    bf16=False
    )

import os
import torch
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
master_port = os.getenv('MASTER_PORT', '12468')
init_method += master_ip + ':' + master_port
torch.distributed.init_process_group(
        backend='nccl',
        world_size=swiss_args.world_size, rank=swiss_args.rank, init_method=init_method)
import sat.mpu as mpu
mpu.initialize_model_parallel(swiss_args.model_parallel_size)
from sat.model.official.yolos_model import YOLOS
swiss_model, swiss_args = YOLOS.from_pretrained(swiss_args, 'yolos-tiny')
swiss_model.get_mixin('pos_embedding').reinit() # patch_embedding should not reinit for inference
model = swiss_model

root = Path(args.coco_path)
assert root.exists(), f'provided COCO path {root} does not exist'
mode = 'instances'
image_set=args.image_set
PATHS = {
    "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
}
img_folder, ann_file = PATHS[image_set]
dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, None), return_masks=False)
img_data, img_anno = dataset.__getitem__(args.index)
ret=nested_tensor_from_tensor_list(img_data.unsqueeze(0))

# device = torch.device("cuda")
device = torch.device("cpu")
model = model.eval()
model.to(device)
ret = ret.to(device)


images = ret.tensors
batch_size, _, height, width = images.shape
num_patches = (height//16) * (width//16)
seq_len = 1 + num_patches + model.get_mixin('det_head').num_det_tokens
position_ids = torch.cat([torch.arange(seq_len)[None,]]*batch_size)
encoded_input = {'input_ids':torch.cat([torch.arange(1+model.get_mixin('det_head').num_det_tokens)[None,]]*batch_size).long(), 'image':images, 'position_ids':position_ids}
encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
encoded_input['attention_mask'] = None

outputs = model(**encoded_input, offline=False, height=height//16, width=width//16)[0]

# outputs = yolos(ret)

# attention = model.forward_return_attention(ret)
# attention = attention[-1].detach().cpu()
# nh = attention.shape[1] # number of head
# attention = attention[0, :, -args.det_token_num:, 1:-args.det_token_num]
#forward input to get pred
result_dic = outputs
# get visualize dettoken index
probas = result_dic['pred_logits'].softmax(-1)[0, :, :-1].cpu()
keep = probas.max(-1).values > 0.9
vis_indexs = torch.nonzero(keep).squeeze(1)
# save original image
os.makedirs(args.output_dir, exist_ok=True)
img = ret.tensors.squeeze(0).cpu()
torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))

# save pred image
save_pred_fig(args.output_dir, result_dic, keep)

# save gt image
save_gt_fig(args.output_dir, img_anno)