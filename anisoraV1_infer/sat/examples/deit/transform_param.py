"""
Given a config file, transform a pretrained ViTModel.
"""
import os

import argparse

parser = argparse.ArgumentParser(description='Set vit model type.')
parser.add_argument('--model', type=str)
model_args = parser.parse_args()
model_type = model_args.model

if model_type == 'tiny':
    from config.deit_tiny_config import vit, args
elif model_type == 'small':
    from config.deit_small_config import vit, args
elif model_type == 'base':
    from config.deit_base_config import vit, args
else:
    raise Exception("Unknown model type. You may see https://github.com/facebookresearch/deit/blob/main/models.py for more model configs")

import torch
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
master_port = os.getenv('MASTER_PORT', '16666')
init_method += master_ip + ':' + master_port
torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank, init_method=init_method)

import sat.mpu as mpu
mpu.initialize_model_parallel(args.model_parallel_size)

from sat.model.official.vit_model import ViTModel
model = ViTModel(args, layernorm_epsilon=1e-6)

def copy_layer_param(src, dst):
    """
    in-place copy from src to dst
    src and dst should be the same layer type, e.g., both are LayerNorm or both are Linear.
        Or at least, both have same named_parameters name and shape.
    """
    src_dic = dict(src.named_parameters())
    dst_dic = dict(dst.named_parameters())
    for k in dst_dic:
        assert dst_dic[k].data.shape == src_dic[k].data.shape
        dst_dic[k].data = src_dic[k].data
        assert (dst_dic[k].data == src_dic[k].data).all()

def copy_from_param(src, dst):
    assert src.data.shape == dst.data.shape
    dst.data = src.data

def copy_layer_norm(src, dst):
    src_ln = []
    for k, v in src.named_parameters():
        if 'norm' in k.lower() and type(v) is not torch.nn.Identity():
            src_ln.append((k, v))
    dst_ln = []
    for k, v in dst.named_parameters():
        if 'layernorm' in k.lower():
            dst_ln.append((k, v))
    assert len(src_ln) == len(dst_ln)
    for kvs, kvd in zip(src_ln, dst_ln):
        assert kvd[1].data.shape == kvs[1].data.shape
        kvd[1].data = kvs[1].data
        assert (kvd[1].data == kvs[1].data).all()

def copy_transformer_layer_wo_ln(src, dst):
    new_weight = src.attn.qkv.weight.data
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = src.attn.qkv.bias.data
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attn.proj, dst.attention.dense)
    copy_layer_param(src.mlp.fc1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.fc2, dst.mlp.dense_4h_to_h)

def transform_weight(src_model, swiss_model):
    copy_from_param(src_model.cls_token.data[0], swiss_model.transformer.word_embeddings.weight)
    copy_from_param(src_model.pos_embed.data[0], swiss_model.transformer.position_embeddings.weight)
    copy_layer_norm(src_model, swiss_model)
    for src_l, dst_l in zip(src_model.blocks, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(src_model.head, swiss_model.mixins.cls.classifier)
    copy_layer_param(src_model.patch_embed.proj, swiss_model.mixins.patch_embedding.proj)
    

vit.eval()
model.eval()
with torch.no_grad():
    transform_weight(vit, model)
    position_ids = torch.cat([torch.arange(197)[None,], torch.arange(197)[None,]])
    encoded_input = {'input_ids':torch.zeros(2, 197).long(), 'image':torch.randn(2, 3, 224, 224)*10, 'position_ids':position_ids}
    src_output = vit(encoded_input['image'])
    model = model.cuda()
    encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
    encoded_input['attention_mask'] = None
    dst_output = model(offline=True, **encoded_input)[0].cpu()
    print("max error:", (src_output - dst_output).abs().max())
    print("max relative error:", ((src_output - dst_output).abs() / torch.max(src_output.abs(), dst_output.abs())).max())
    torch.save({'module':model.state_dict()}, "output.pt")

breakpoint()
