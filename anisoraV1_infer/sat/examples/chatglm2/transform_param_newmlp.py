from sat.model import ChatGLM2Model
from transformers import AutoTokenizer, AutoModel, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
chatglm = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

import argparse
args = argparse.Namespace(
    num_layers=config.num_layers,
    vocab_size=config.padded_vocab_size,
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    max_sequence_length=config.seq_length,
    hidden_dropout=0.,
    attention_dropout=0.,
    inner_hidden_size=config.ffn_hidden_size,
    num_multi_query_heads=config.multi_query_group_num,
    is_gated_mlp=True,
    layernorm_epsilon=config.layernorm_epsilon,
    hidden_size_per_attention_head=None,
    use_bias=False,
    use_qkv_bias=True,
    checkpoint_activations=None,
    layernorm_order='pre',
    model_parallel_size=1,
    world_size=1,
    rank=0,
    skip_init=True,
    use_gpu_initialization=False,
    save='chatglm2-6b',
    deepspeed=None,
    mode='inference',
    tokenizer_type="THUDM/chatglm2-6b"
    )

model = ChatGLM2Model(args).half()

def copy_layer_param(src, dst):
    """
    in-place copy from src to dst
    src and dst should be the same layer type, e.g., both are LayerNorm or both are Linear.
        Or at least, both have same named_parameters name and shape.
    """
    src_dic = dict(src.named_parameters())
    dst_dic = dict(dst.named_parameters())
    for k in dst_dic:
        if dst_dic[k].data.shape != src_dic[k].data.shape:
            print(dst_dic[k].data.shape, src_dic[k].data.shape)
        assert dst_dic[k].data.shape == src_dic[k].data.shape
        dst_dic[k].data = src_dic[k].data
        assert (dst_dic[k].data == src_dic[k].data).all()

def copy_transformer_layer(src, dst):
    copy_layer_param(src.self_attention.query_key_value, dst.attention.query_key_value)
    copy_layer_param(src.self_attention.dense, dst.attention.dense)
    weight_1, weight_2 = src.mlp.dense_h_to_4h.weight.data.chunk(2, dim=0)
    assert dst.mlp.dense_h_to_4h_gate.weight.data.shape == weight_1.shape
    dst.mlp.dense_h_to_4h_gate.weight.data = weight_1
    assert dst.mlp.dense_h_to_4h.weight.data.shape == weight_2.shape
    dst.mlp.dense_h_to_4h.weight.data = weight_2
    # copy_layer_param(src.mlp.dense_h_to_4h, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.mlp.dense_4h_to_h, dst.mlp.dense_4h_to_h)
    copy_layer_param(src.input_layernorm, dst.input_layernorm)
    copy_layer_param(src.post_attention_layernorm, dst.post_attention_layernorm)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.transformer.embedding.word_embeddings, swiss_model.transformer.word_embeddings)
    for src_l, dst_l in zip(hugging_model.transformer.encoder.layers, swiss_model.transformer.layers):
        copy_transformer_layer(src_l, dst_l)
    copy_layer_param(hugging_model.transformer.encoder.final_layernorm, swiss_model.transformer.final_layernorm)
    copy_layer_param(hugging_model.transformer.output_layer, model.mixins['chatglm-final'].lm_head)

import torch
from sat.training.model_io import save_checkpoint
chatglm.eval()
model.eval()
with torch.no_grad():
    transform_weight(chatglm, model)
    save_checkpoint(1, model, None, None, args)
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input = {k:v.cuda() for k, v in encoded_input.items()}
    hugging_output = chatglm(**encoded_input).logits.cpu()
    dst_output = model.cuda()(input_ids=encoded_input['input_ids'], position_ids=encoded_input['position_ids'], attention_mask=torch.ones(1, 1, dtype=torch.float16, device='cuda'))
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())

breakpoint()