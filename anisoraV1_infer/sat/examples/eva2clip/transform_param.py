from sat.model.official import EVA2CLIPModel
import torch
import torch.nn as nn
from eva2_clip import load_checkpoint
from eva2_clip import get_model_config
from eva2_clip import CustomCLIP

class Eva2Encoder(nn.Module):
    def __init__(self, image_size=224, ckpt_path=''):
        super(Eva2Encoder, self).__init__()
        self.config = get_model_config('EVA02-CLIP-bigE-14')
        self.config['vision_cfg']['image_size'] = image_size
        model = CustomCLIP(**self.config)
        load_checkpoint(model, ckpt_path)
        self.model = model.visual

    def forward(self, **kwargs):
        encode = self.model(kwargs['image'], return_all_features=True)[:, 1:, :]
        return encode

model = Eva2Encoder(image_size=224, ckpt_path='EVA02_CLIP_E_psz14_s4B.pt').bfloat16().cuda()

import argparse
tmp_args = argparse.Namespace(
    num_layers=63,
    vocab_size=1,
    hidden_size=1792,
    num_attention_heads=16,
    num_multi_query_heads=0,
    in_channels=3,
    image_size=[224, 224],
    patch_size=14,
    pre_len=1,
    post_len=0,
    layernorm_epsilon=1e-6,
    use_bias=True,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=int(1792*8.571428571428571),
    use_final_layernorm=False,
    row_parallel_linear_final_bias=False,
    hidden_size_per_attention_head=None,
    layernorm_order='post',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=True,
    use_gpu_initialization=None,
    save='eva-clip-4b-14-x-drop-last-layer',
    deepspeed=None,
    mode='inference',
    tokenizer_type='fake',
    bf16=True,
    fp16=False
    )

new_vit_model = EVA2CLIPModel(args=tmp_args)
new_vit_model.eval()

def copy_from_param(src, dst):
    assert src.data.shape == dst.data.shape
    dst.data = src.data

def copy_layer_param(src, dst):
    """
    in-place copy from src to dst
    src and dst should be the same layer type, e.g., both are LayerNorm or both are Linear.
        Or at least, both have same named_parameters name and shape.
    """
    src_dic = dict(src.named_parameters())
    dst_dic = dict(dst.named_parameters())
    if set(src_dic.keys()) != set(dst_dic.keys()):
        breakpoint()
    for k in dst_dic:
        assert dst_dic[k].data.shape == src_dic[k].data.shape
        dst_dic[k].data = src_dic[k].data
        assert (dst_dic[k].data == src_dic[k].data).all()

def copy_transformer_layer(src, dst):
    new_weight = src.attn.qkv.weight
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = torch.cat([src.attn.q_bias, torch.zeros_like(src.attn.v_bias), src.attn.v_bias])
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attn.proj, dst.attention.dense)
    copy_layer_param(src.mlp.fc1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.fc2, dst.mlp.dense_4h_to_h)
    copy_layer_param(src.norm1, dst.input_layernorm)
    copy_layer_param(src.norm2, dst.post_attention_layernorm)

def transform_weight(hugging_model, swiss_model):
    copy_from_param(hugging_model.cls_token.data[0], swiss_model.transformer.word_embeddings.weight)
    copy_from_param(hugging_model.pos_embed.data[0], swiss_model.transformer.position_embeddings.weight)
    for src_l, dst_l in zip(hugging_model.blocks, swiss_model.transformer.layers):
        copy_transformer_layer(src_l, dst_l)
    copy_layer_param(hugging_model.patch_embed.proj, swiss_model.mixins.patch_embedding.proj)

from sat.training.model_io import save_checkpoint
from transforms import BlipImageEvalProcessor
from functools import partial

def blip2_image_processor_func(image_processor, image):
    return {'image': image_processor(image).unsqueeze(0).bfloat16()}

def blip2_image_processor_func_with_inputs(image_processor, image):
    return {'image': image_processor(image).unsqueeze(0).bfloat16(), 'input_ids': torch.zeros(1, 1, dtype=torch.long), 'position_ids': None, 'attention_mask': torch.ones(1, 1, dtype=torch.long)}

blip2_image_processor = partial(blip2_image_processor_func, BlipImageEvalProcessor(224))
blip2_image_processor_sat = partial(blip2_image_processor_func_with_inputs, BlipImageEvalProcessor(224))
from PIL import Image

with torch.no_grad():
    transform_weight(model.model, new_vit_model)
    img = Image.open("kobe.png").convert("RGB")
    inputs1 = blip2_image_processor(img)
    inputs1 = {k:inputs1[k].cuda() if inputs1[k] is not None else None for k in inputs1}
    output1 = model(**inputs1)
    inputs2 = blip2_image_processor_sat(img)
    inputs2 = {k:inputs2[k].cuda() if inputs2[k] is not None else None  for k in inputs2}
    output2 = new_vit_model(**inputs2)
    save_checkpoint(1, new_vit_model, None, None, tmp_args)
    breakpoint()