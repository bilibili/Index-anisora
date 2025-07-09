from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModel, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
chatglm = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).cuda()
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

import argparse
args = argparse.Namespace(
    num_layers=config.num_layers,
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    max_sequence_length=config.max_sequence_length,
    bos_token_id=tokenizer.bos_token_id,
    mask_token_id=tokenizer.mask_token_id,
    gmask_token_id=tokenizer.gmask_token_id,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=True,
    checkpoint_num_layers=1,
    layernorm_order='post',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=False,
    use_gpu_initialization=True,
    save='chatglm-6b',
    deepspeed=None,
    mode='inference',
    tokenizer_type="THUDM/chatglm-6b"
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
from sat.model import ChatGLMModel

model = ChatGLMModel(args)


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
        if 'layernorm' in k.lower():
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
    new_weight = src.attention.query_key_value.weight.data
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = src.attention.query_key_value.bias.data
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attention.dense, dst.attention.dense)
    copy_layer_param(src.mlp.dense_h_to_4h, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.dense_4h_to_h, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.transformer.word_embeddings, swiss_model.transformer.word_embeddings)
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l in zip(hugging_model.transformer.layers, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(hugging_model.lm_head, model.mixins['chatglm-final'].lm_head)

from sat.training.model_io import save_checkpoint
chatglm.eval()
model.eval()
with torch.no_grad():
    transform_weight(chatglm, model)
    save_checkpoint(1, model, None, None, args)
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input = {k:v.cuda() for k, v in encoded_input.items()}
    hugging_output = chatglm.half()(**encoded_input).logits.cpu()
    dst_output = model.half().cuda()(input_ids=encoded_input['input_ids'])
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())

breakpoint()