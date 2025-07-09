from asuka.modeling_pretrain import eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init
import huggingface_hub
import torch

path = huggingface_hub.hf_hub_download('Yuxin-CV/EVA-02', 'eva02_L_pt_m38m_p14.pt', subfolder='eva02/pt')
print(path)
eva = eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init(predict_feature_dim=1024)
state_dict = torch.load(path, map_location="cpu")["module"]
for k in list(state_dict.keys()):
    if 'freqs_cos' in k or 'freqs_sin' in k:
        del state_dict[k]
for k in state_dict:
    dtype = state_dict[k].dtype
    if dtype != torch.float16 and dtype != torch.float32:
        raise Exception("???")
    else:
        if dtype == torch.float16:
            eva = eva.half()
    break
eva.load_state_dict(state_dict, strict=False)
# x = torch.randn(2, 3, 224, 224).half().cuda()
# bool_mask = torch.ones(2, 256, dtype=torch.bool).cuda()
# bool_mask[:, 1] = False
# out = eva(x, bool_mask)

import argparse
args = argparse.Namespace(
    image_size=[224, 224],
    patch_size=14,
    pre_len=1,
    post_len=0,
    in_channels=3,
    predict_feature_dim=1024,
    num_layers=24,
    vocab_size=1,
    hidden_size=1024,
    num_attention_heads=16,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=int(1024 * (4*2/3)),
    hidden_size_per_attention_head=None,
    checkpoint_activations=True,
    checkpoint_num_layers=1,
    layernorm_order='pre',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=False,
    use_gpu_initialization=True,
    save='eva02_L_pt_m38m_p14',
    deepspeed=None,
    mode='inference',
    tokenizer_type="Fake"
    )

import os
os.makedirs(args.save, exist_ok=True)
import torch
import deepspeed
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
master_port = os.getenv('MASTER_PORT', '16666')
init_method += master_ip + ':' + master_port
torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank, init_method=init_method)
deepspeed.init_distributed(
        dist_backend='nccl',
        world_size=args.world_size, rank=args.rank, init_method=init_method)

import sat.mpu as mpu
mpu.initialize_model_parallel(args.model_parallel_size)
from sat.model import EVA2Model

model = EVA2Model(args, layernorm_epsilon=1e-6)

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

def copy_layer_norm(src, dst):
    src_ln = []
    for k, v in src.named_parameters():
        if 'norm' in k.lower():
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
    src_ln = []
    for k, v in src.named_parameters():
        if 'ffn_ln' in k.lower():
            src_ln.append((k, v))
    dst_ln = []
    for k, v in dst.named_parameters():
        if 'ffn_ln' in k.lower():
            dst_ln.append((k, v))
    assert len(src_ln) == len(dst_ln)
    for kvs, kvd in zip(src_ln, dst_ln):
        assert kvd[1].data.shape == kvs[1].data.shape
        kvd[1].data = kvs[1].data
        assert (kvd[1].data == kvs[1].data).all()

def copy_transformer_layer_wo_ln(src, dst, w2):
    new_weight = torch.cat([src.attn.q_proj.weight.data, src.attn.k_proj.weight.data, src.attn.v_proj.weight.data], 0)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    k_bias = torch.zeros_like(src.attn.q_bias.data)
    new_bias = torch.cat([src.attn.q_bias.data, k_bias, src.attn.v_bias.data], 0)
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attn.proj, dst.attention.dense)
    copy_layer_param(src.mlp.w2, w2)
    copy_layer_param(src.mlp.w1, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.w3, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    swiss_model.mixins['patch_embedding'].mask_token.data = hugging_model.mask_token.data
    swiss_model.transformer.word_embeddings.weight.data = hugging_model.cls_token.data[0]
    copy_layer_param(hugging_model.patch_embed.proj, swiss_model.mixins['patch_embedding'].proj)
    swiss_model.transformer.position_embeddings.weight.data = hugging_model.pos_embed.data[0]
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l, w2 in zip(hugging_model.blocks, swiss_model.transformer.layers, swiss_model.mixins['eva2-mlp'].w2):
        copy_transformer_layer_wo_ln(src_l, dst_l, w2)
    copy_layer_param(hugging_model.lm_head, swiss_model.mixins['eva2-final'].lm_head)

from sat.training.model_io import save_checkpoint
eva.eval().cuda()
model.eval()
with torch.no_grad():
    transform_weight(eva, model)
    model.cuda()
    save_checkpoint(1, model, None, None, args)
    x = torch.randn(2, 3, 224, 224).half().cuda()
    bool_mask = torch.ones(2, 256, dtype=torch.bool).cuda()
    bool_mask[:, 1] = False
    hugging_output = eva(x, bool_mask).cpu()
    input_ids = torch.zeros(2, 1, dtype=torch.long).cuda()
    attention_mask = torch.tensor([[1.]], dtype=torch.float16).cuda()
    dst_output = model(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=x, bool_masked_pos=bool_mask)
    swiss_output = dst_output[0].cpu()
    print(hugging_output)
    print(swiss_output)
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())

breakpoint()