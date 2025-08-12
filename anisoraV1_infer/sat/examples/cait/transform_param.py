"""
Given a config file, transform a pretrained CaiT.
"""
import os

from hjson import OrderedDict
from config.cait_config import vit, args

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

from sat.model.official.cait_model import CaiT
model = CaiT(args, layernorm_epsilon=1e-6)

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
        if 'layernorm' in k.lower() and not ('decoder' in k and 'post_attention' in k):
            dst_ln.append((k, v))
    assert len(src_ln) == len(dst_ln)
    for kvs, kvd in zip(src_ln, dst_ln):
        assert kvd[1].data.shape == kvs[1].data.shape
        kvd[1].data = kvs[1].data
        assert (kvd[1].data == kvs[1].data).all()

def copy_transformer_layer_wo_ln_encoder(src, dst, ind):
    new_weight = src.attn.qkv.weight.data
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = src.attn.qkv.bias.data
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attn.proj, dst.attention.dense)
    copy_layer_param(src.mlp.fc1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.fc2, dst.mlp.dense_4h_to_h)

def copy_mixin_encoder(src, dst, ind):
    copy_layer_param(src.attn.proj_l, dst.encoder.mixins.attn.proj_l[ind])
    copy_layer_param(src.attn.proj_w, dst.encoder.mixins.attn.proj_w[ind])
    copy_from_param(src.gamma_1.data, dst.encoder.mixins.enc_forward.gamma_1[ind])
    copy_from_param(src.gamma_2.data, dst.encoder.mixins.enc_forward.gamma_2[ind])

def copy_transformer_layer_wo_ln_decoder(src, dst, ind):
    dst.cross_attention.query.weight.data = src.attn.q.weight.data
    dst.cross_attention.query.bias.data = src.attn.q.bias.data
    new_weight = torch.cat([src.attn.k.weight.data, src.attn.v.weight.data], 0)
    new_bias = torch.cat([src.attn.k.bias.data, src.attn.v.bias.data], 0)
    assert dst.cross_attention.key_value.weight.data.shape == new_weight.shape
    assert dst.cross_attention.key_value.bias.data.shape == new_bias.shape
    dst.cross_attention.key_value.weight.data = new_weight
    dst.cross_attention.key_value.bias.data = new_bias
    copy_layer_param(src.attn.proj, dst.cross_attention.dense)
    copy_layer_param(src.mlp.fc1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.fc2, dst.mlp.dense_4h_to_h)

def copy_mixin_decoder(src, dst, ind):
    copy_from_param(src.gamma_1.data, dst.decoder.mixins.dec_forward.gamma_1[ind])
    copy_from_param(src.gamma_2.data, dst.decoder.mixins.dec_forward.gamma_2[ind])

def transform_weight(src_model, swiss_model):
    # transform layernorm
    copy_layer_norm(src_model, swiss_model)
    # transform encoder
    copy_from_param(src_model.cls_token.data[0], swiss_model.encoder.transformer.word_embeddings.weight)
    copy_from_param(src_model.pos_embed.data[0], swiss_model.encoder.transformer.position_embeddings.weight)
    for ind, src_l, dst_l in zip(range(len(src_model.blocks)), src_model.blocks, swiss_model.encoder.transformer.layers):
        copy_transformer_layer_wo_ln_encoder(src_l, dst_l, ind)
        copy_mixin_encoder(src_l, swiss_model, ind)
    copy_layer_param(src_model.patch_embed.proj, swiss_model.encoder.mixins.patch_embedding.proj)
    # transform decoder
    copy_from_param(src_model.cls_token.data[0], swiss_model.decoder.transformer.word_embeddings.weight)
    for ind, src_l, dst_l in zip(range(len(src_model.blocks_token_only)), src_model.blocks_token_only, swiss_model.decoder.transformer.layers):
        copy_transformer_layer_wo_ln_decoder(src_l, dst_l, ind)
        copy_mixin_decoder(src_l, swiss_model, ind)
    copy_layer_param(src_model.head, swiss_model.decoder.mixins.cls.classifier)
    

vit.eval()
model.eval()
with torch.no_grad():
    transform_weight(vit, model)
    position_ids = torch.cat([torch.arange(196)[None,], torch.arange(196)[None,]])
    dec_pos = torch.cat([torch.arange(1)[None,], torch.arange(1)[None,]])
    encoded_input = {'input_ids':torch.zeros(2, 1).long(), 'image':torch.randn(2, 3, 224, 224), 'enc_position_ids':position_ids, 'dec_position_ids':dec_pos}
    src_output = vit(encoded_input['image'])
    model = model.cuda()
    encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
    dst_enc_output, dst_output, *_ = model(offline=True, **encoded_input)
    dst_output = dst_output.cpu()
    print("max error:", (src_output - dst_output).abs().max())
    print("max relative error:", ((src_output - dst_output).abs() / torch.max(src_output.abs(), dst_output.abs())).max())
    torch.save({'module':model.state_dict()}, "output.pt")

breakpoint()
