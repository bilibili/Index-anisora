import os
pretrain_path = '/data/qingsong/pretrain'
model_type = 'bert-large-uncased'

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrain_path, model_type))
bert = BertForMaskedLM.from_pretrained(os.path.join(pretrain_path, model_type), output_hidden_states=True)
lm_head = bert.cls
bert = bert.bert

import argparse
if model_type == 'bert-base-uncased':
    args = argparse.Namespace(
        num_layers=12,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        max_sequence_length=512,
        num_types=2,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        layernorm_order='post',
        model_parallel_size=1,
        world_size=1,
        rank=0
        )
elif model_type == 'bert-large-uncased':
    args = argparse.Namespace(
        num_layers=24,
        vocab_size=30522,
        hidden_size=1024,
        num_attention_heads=16,
        max_sequence_length=512,
        num_types=2,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        layernorm_order='post',
        model_parallel_size=1,
        world_size=1,
        rank=0
        )
else:
    raise Exception("model type not recognized!")

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

from sat.model.official.bert_model import BertModel
model = BertModel(args, layernorm_epsilon=1e-12)

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
    new_weight = torch.cat([src.attention.self.query.weight.data, src.attention.self.key.weight.data, src.attention.self.value.weight.data], 0)
    assert dst.attention.query_key_value.weight.data.shape == new_weight.shape
    dst.attention.query_key_value.weight.data = new_weight
    new_bias = torch.cat([src.attention.self.query.bias.data, src.attention.self.key.bias.data, src.attention.self.value.bias.data], 0)
    assert dst.attention.query_key_value.bias.data.shape == new_bias.shape
    dst.attention.query_key_value.bias.data = new_bias
    copy_layer_param(src.attention.output.dense, dst.attention.dense)
    copy_layer_param(src.intermediate.dense, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.output.dense, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.embeddings.word_embeddings, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.embeddings.position_embeddings, swiss_model.transformer.position_embeddings)
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l in zip(hugging_model.encoder.layer, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)
    copy_layer_param(lm_head.predictions.transform.dense, model.mixins['bert-final'].lm_head.dense)
    copy_layer_param(lm_head.predictions.transform.LayerNorm, model.mixins['bert-final'].lm_head.layer_norm)
    copy_layer_param(lm_head.predictions.decoder, model.mixins['bert-final'].lm_head.decoder)
    copy_layer_param(hugging_model.embeddings.token_type_embeddings, swiss_model.mixins['bert-type'].type_embeddings)


bert.eval()
model.eval()
with torch.no_grad():
    transform_weight(bert, model)
    text = [["This is a piece of text.", "Another piece of text."]]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    print(position_ids)
    output = bert(**encoded_input)
    hugging_output = lm_head(output[0])
    model.to('cuda:0')
    dst_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output[:,0] - swiss_output[:,0]).abs().max())
    print("max relative error:", ((hugging_output[:,0] - swiss_output[:,0]).abs() / torch.max(swiss_output[:,0].abs(), hugging_output[:,0].abs())).max())
    torch.save({'module':model.state_dict()}, "output.pt")

# breakpoint()