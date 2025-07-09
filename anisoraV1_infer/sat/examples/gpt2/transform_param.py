# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# print(result)
model_type = 'gpt2'


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

gpt2 = GPT2LMHeadModel.from_pretrained(model_type)
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
config = GPT2Config.from_pretrained(model_type)
import argparse
args = argparse.Namespace(
    num_layers=config.n_layer,
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    num_attention_heads=config.n_head,
    max_sequence_length=config.max_position_embeddings,
    hidden_dropout=config.embd_pdrop,
    attention_dropout=config.attn_pdrop,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=True,
    checkpoint_num_layers=1,
    layernorm_order='pre',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=False,
    use_gpu_initialization=True,
    save='sat_'+model_type,
    deepspeed=None,
    mode='inference',
    tokenizer_type=model_type
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
from sat.model import GPT2Model

model = GPT2Model(args)


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
        if 'ln' in k.lower():
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

def copy_linear_to_conv1d(src, dst):
    dst.weight.data = src.weight.data.transpose(0, 1)
    dst.bias.data = src.bias.data

def copy_transformer_layer_wo_ln(src, dst):
    new_weight = src.attn.c_attn.weight.data.transpose(0, 1)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = src.attn.c_attn.bias.data
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_linear_to_conv1d(src.attn.c_proj, dst.attention.dense)
    copy_linear_to_conv1d(src.mlp.c_fc, dst.mlp.dense_h_to_4h)
    copy_linear_to_conv1d(src.mlp.c_proj, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.transformer.wte, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.transformer.wpe, swiss_model.transformer.position_embeddings)
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l in zip(hugging_model.transformer.h, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(hugging_model.lm_head, model.mixins['gpt2-final'].lm_head)

from sat.training.model_io import save_checkpoint
gpt2.eval()
model.eval()
with torch.no_grad():
    transform_weight(gpt2, model)
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt')
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    print(position_ids)
    hugging_output = gpt2(**encoded_input).logits
    model.to('cuda:0')
    dst_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())
    # torch.save({'module':model.state_dict()}, model_type+".pt")
    save_checkpoint(1, model, None, None, args)


breakpoint()