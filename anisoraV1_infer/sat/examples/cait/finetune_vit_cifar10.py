# -*- encoding: utf-8 -*-
# @File    :   finetune_vit_cifar10.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

# here put the import lib
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sat import mpu, get_args
from cait_ft_model import CaiTFinetuneModel
from sat.model.official.cait_model import CaiTEncoder
from sat.training.deepspeed_training import training_main


def get_batch(data_iterator, args, timers):
    # Items and their type.

    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    image_data = {"image":data[0]}
    label_data = {"label":data[1]}
    timers('data loader').stop()
    image_data = mpu.broadcast_data(["image"], image_data, torch.float32)
    label_data = mpu.broadcast_data(["label"], label_data, torch.int64)

    # Unpack.
    label_data = label_data['label'].long()
    image_data = image_data['image']
    batch_size = label_data.size()[0]
    seq_length = args.pre_len + (args.image_size[0]//args.patch_size)*(args.image_size[1]//args.patch_size) + args.post_len
    position_ids = torch.zeros(seq_length, device=image_data.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[:seq_length])
    position_ids = position_ids.unsqueeze(0).expand([batch_size, -1])
    attention_mask = torch.ones((1, 1), device=image_data.device)

    tokens = torch.zeros((batch_size, 1), device=image_data.device, dtype=torch.long)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
        image_data = image_data.half()
    return tokens, image_data, label_data, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, images, labels, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)

    timers('batch generator').stop()

    encoder_outputs, decoder_outputs, *mems = model(tokens, position_ids, attention_mask, image=images, offline=True) #, offline=False, height=384//16, width=384//16)
    loss = F.cross_entropy(decoder_outputs, labels)
    acc = (torch.argmax(decoder_outputs, dim=-1) == labels).sum() / labels.numel()
    return loss, {'acc': acc}

#/dataset/fd5061f6/satDatasets/
def create_dataset_function(path, args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(384),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/'.join(path.split('/')[:-1]), train=(path.split('/')[-1]=='train'),
                                            download=True, transform=transform)
    return trainset

def init_function(args, model):
    model.encoder.get_mixin("pos_embedding").reinit()

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--md_type', type=str)
    # py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser = CaiTFinetuneModel.add_model_specific_args(py_parser)
    py_parser = CaiTEncoder.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from sat.training.deepspeed_training import initialize_distributed, set_random_seed
    # initialize_distributed(args)
    # set_random_seed(args.seed)
    model, args = CaiTFinetuneModel.from_pretrained(args.md_type, args)
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, init_function=init_function)
