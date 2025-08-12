"""
Given a config file, transform a pretrained MAE model.
"""
import os

from hjson import OrderedDict
from psutil import virtual_memory
from config.clip_config import vit, args, processor

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

from sat.model.official.clip_model import CLIP
model = CLIP(args, layernorm_epsilon=1e-5)

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
    new_weight = torch.cat([src.self_attn.q_proj.weight, src.self_attn.k_proj.weight, src.self_attn.v_proj.weight])
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight.data
    new_bias = torch.cat([src.self_attn.q_proj.bias, src.self_attn.k_proj.bias, src.self_attn.v_proj.bias]) 
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias.data
    copy_layer_param(src.self_attn.out_proj, dst.attention.dense)
    copy_layer_param(src.mlp.fc1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.fc2, dst.mlp.dense_4h_to_h)

def transform_image_weight(src_model, swiss_model):
    copy_layer_norm(src_model, swiss_model)
    copy_from_param(src_model.embeddings.class_embedding.unsqueeze(0).data, swiss_model.transformer.word_embeddings.weight)
    copy_layer_param(src_model.embeddings.position_embedding, swiss_model.transformer.position_embeddings)
    for src_l, dst_l in zip(src_model.encoder.layers, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(src_model.embeddings.patch_embedding, swiss_model.mixins.patch_embedding.proj)

def transform_text_weight(src_model, swiss_model):
    copy_layer_norm(src_model, swiss_model)
    copy_layer_param(src_model.embeddings.token_embedding, swiss_model.transformer.word_embeddings)
    copy_layer_param(src_model.embeddings.position_embedding, swiss_model.transformer.position_embeddings)
    for src_l, dst_l in zip(src_model.encoder.layers, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)

def transform_both_weight(src_model, swiss_model):
    transform_image_weight(src_model.vision_model, swiss_model.image_encoder)
    transform_text_weight(src_model.text_model, swiss_model.text_encoder)
    copy_layer_param(src_model.visual_projection, swiss_model.image_encoder.mixins.image_enc.visual_projection)
    copy_layer_param(src_model.text_projection, swiss_model.text_encoder.mixins.text_enc.text_projection)
    copy_from_param(src_model.logit_scale, swiss_model.logit_scale)

from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)
expanded_mask = inputs['attention_mask'][:, None, None, :].expand(2, 1, 7, 7).to(torch.float)
vit.eval()
model.eval()
with torch.no_grad():
    transform_both_weight(vit, model)
    image_position_ids = torch.cat([torch.arange(50)[None,]])
    text_position_ids = torch.cat([torch.arange(7)[None,], torch.arange(7)[None,]])
    encoded_input = {'text_attention_mask':expanded_mask, 'image_input_ids':torch.zeros(1, 1).long(), 'image_position_ids':image_position_ids, 'image':inputs['pixel_values'], 'text_input_ids':inputs['input_ids'], 'text_position_ids':text_position_ids}
    src_output = vit(**inputs, output_hidden_states=True)
    print(src_output.logits_per_text)
    model = model.cuda()
    encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
    image_embeds, text_embeds, logits_per_text, logits_per_image = model(offline=True, **encoded_input)
    logits_per_text = logits_per_text.cpu()
    print(logits_per_text)
    src_output = src_output.logits_per_text
    dst_output = logits_per_text
    print("max error:", (src_output - dst_output).abs().max())
    print("max relative error:", ((src_output - dst_output).abs() / torch.max(src_output.abs(), dst_output.abs())).max())
    torch.save({'module':model.state_dict()}, "output.pt")

breakpoint()
