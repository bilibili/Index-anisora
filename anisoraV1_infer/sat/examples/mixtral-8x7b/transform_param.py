from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

hugging = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()
config = AutoConfig.from_pretrained(model_id)

from sat.model import MixtralModel
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
    layernorm_epsilon=config.rms_norm_eps,
    use_bias=False,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=config.intermediate_size,
    hidden_size_per_attention_head=None,
    layernorm_order='pre',
    num_experts=8,
    num_experts_per_tok=2,
    is_gated_mlp=True,
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=True,
    use_gpu_initialization=None,
    save="mixtral-8x7b-instruct",
    deepspeed=None,
    mode='inference',
    tokenizer_type="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

model = MixtralModel(args=args).float()
model.eval()

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
        assert dst_dic[k].data.shape == src_dic[k].data.shape, f"src {src_dic[k].data.shape}, dst {dst_dic[k].data.shape}"
        dst_dic[k].data = src_dic[k].data
        assert (dst_dic[k].data == src_dic[k].data).all()

def copy_transformer_layer(src, dst, gate):
    new_weight = torch.cat([src.self_attn.q_proj.weight.data, src.self_attn.k_proj.weight.data, src.self_attn.v_proj.weight.data], 0)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    copy_layer_param(src.self_attn.o_proj, dst.attention.dense)
    copy_layer_param(src.block_sparse_moe.gate, gate)
    for i in range(len(src.block_sparse_moe.experts)):
        suffix = f"_{i}" if i > 0 else ""
        copy_layer_param(src.block_sparse_moe.experts[i].w1, getattr(dst.mlp, "dense_h_to_4h_gate"+suffix))
        copy_layer_param(src.block_sparse_moe.experts[i].w3, getattr(dst.mlp, "dense_h_to_4h"+suffix))
        copy_layer_param(src.block_sparse_moe.experts[i].w2, getattr(dst.mlp, "dense_4h_to_h"+suffix))
    copy_layer_param(src.input_layernorm, dst.input_layernorm)
    copy_layer_param(src.post_attention_layernorm, dst.post_attention_layernorm)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.model.embed_tokens, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.lm_head, swiss_model.get_mixin("lm").lm_head)
    for src_l, dst_l, gate in zip(hugging_model.model.layers, swiss_model.transformer.layers, swiss_model.get_mixin("mlp").gates):
        copy_transformer_layer(src_l, dst_l, gate)
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
    out_h = hugging(**batch)
    batch['attention_mask'] = batch['attention_mask'].unsqueeze(1).repeat_interleave(batch['attention_mask'].shape[-1], 1).tril().unsqueeze(1)
    out_s = model(**batch)
    breakpoint()