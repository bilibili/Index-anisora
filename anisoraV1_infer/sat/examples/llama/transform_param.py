from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

prefix = 'meta-llama/'
model_type = 'Llama-2-70b-chat-hf'

config = LlamaConfig.from_pretrained(prefix+model_type)
tokenizer = LlamaTokenizer.from_pretrained(prefix+model_type)
hugging = LlamaForCausalLM.from_pretrained(prefix+model_type).eval().half()
if '70b' not in model_type:
    hugging = hugging.cuda()

from sat.model import LLaMAModel
import torch

import argparse
args = argparse.Namespace(
    num_layers=config.num_hidden_layers,
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_multi_query_heads=0 if config.num_attention_heads == config.num_key_value_heads else config.num_key_value_heads,
    max_sequence_length=config.max_position_embeddings,
    bos_token_id=config.bos_token_id,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id,
    layernorm_epsilon=config.rms_norm_eps,
    is_rotary_emb=True,
    is_gated_mlp=True,
    use_bias=False,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=config.intermediate_size,
    hidden_size_per_attention_head=None,
    layernorm_order='pre',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=True,
    use_gpu_initialization=None,
    save=model_type[:-3].lower(),
    deepspeed=None,
    mode='inference',
    tokenizer_type=model_type
    )

model = LLaMAModel(args=args)
model.eval()
if '70b' not in model_type:
    if args.is_rotary_emb:
        model.transformer.position_embeddings.cuda()
    else:
        model.get_mixin("rotary").cuda()

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
    new_weight = torch.cat([src.self_attn.q_proj.weight.data, src.self_attn.k_proj.weight.data, src.self_attn.v_proj.weight.data], 0)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    copy_layer_param(src.self_attn.o_proj, dst.attention.dense)
    copy_layer_param(src.mlp.up_proj, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.gate_proj, dst.mlp.dense_h_to_4h_gate)
    copy_layer_param(src.mlp.down_proj, dst.mlp.dense_4h_to_h)
    copy_layer_param(src.input_layernorm, dst.input_layernorm)
    copy_layer_param(src.post_attention_layernorm, dst.post_attention_layernorm)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.model.embed_tokens, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.lm_head, swiss_model.get_mixin("lm").lm_head)
    for src_l, dst_l in zip(hugging_model.model.layers, swiss_model.transformer.layers):
        copy_transformer_layer(src_l, dst_l)
    copy_layer_param(hugging_model.model.norm, swiss_model.transformer.final_layernorm)

from sat.training.model_io import save_checkpoint

with torch.no_grad():
    transform_weight(hugging, model)
    save_checkpoint(1, model, None, None, args)

    batch = tokenizer(
        "The capital of China is",
        return_tensors="pt", 
        add_special_tokens=False
    )
    batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).unsqueeze(0)
    if '70b' not in model_type:
        batch = {k: v.cuda() for k, v in batch.items()}
    out_h = hugging(**batch)
    batch['attention_mask'] = batch['attention_mask'].unsqueeze(1).repeat_interleave(batch['attention_mask'].shape[-1], 1).tril().unsqueeze(1)
    out_s = model(**batch)
    breakpoint()