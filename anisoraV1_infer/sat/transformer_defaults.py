# coding=utf-8
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_defaults.py
@Time    :   2022/06/01 21:44:17
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import math
import torch#,torch_npu
import torch.distributed
import torch.nn.functional as F

from sat import mpu

from sat.mpu.utils import split_tensor_along_last_dim
import contextlib

def standard_attention(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
        # if auto-regressive, skip
        attention_scores = torch.mul(attention_scores, attention_mask) - \
                           10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)

    if attention_dropout is not None:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = attention_dropout(attention_probs)
        else:
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # expand head dim to query dim, if necessary
    # only useful for multi-query attention
    batch_size, num_query_heads = query_layer.shape[:2] # [b, np, s, hn]
    num_kv_heads = key_layer.shape[1] # [b, np, s, hn]
    key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
    value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])

    is_low_triangle = (attention_mask == torch.ones_like(attention_mask, dtype=torch.float).tril()).all()
    is_full = (attention_mask is None) or (attention_mask > 0).all()

    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        if dropout_p > 0 and mpu.get_cuda_rng_tracker is not None:
            context = mpu.get_cuda_rng_tracker().fork()
        else:
            context = contextlib.nullcontext()
        with context:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=not is_full
            )

            # attn_output=torch_npu.npu_fusion_attention(
            #     query_layer,key_layer,value_layer,num_query_heads,input_layout="BNSD",pse=None,atten_mask=None,
            #     scale=1.0/math.sqrt(query_layer.shape[-1]),
            #     pre_tockens=65536,
            #     next_tockens=65536,
            #     keep_prob=1.,
            #     sync=False,
            #     inner_precise=0
            # )[0]
        return attn_output
    else:
        return standard_attention(
            query_layer, key_layer, value_layer, attention_mask,
            attention_dropout=attention_dropout, log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score, **kwargs
        )

from sat.sequence_parallel.comm import (
    all_to_all_with_pad,
    gather_sequence,
    get_spatial_pad,
    get_temporal_pad,
    set_spatial_pad,
    set_temporal_pad,
    split_sequence,
)

def attention_forward_default(self, hidden_states, mask, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_raw_layer = self.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

    dropout_fn = self.attention_dropout if self.training else None

    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)
    # rotary position embedding 
    if self.transformer.is_rotary_emb:
        query_layer, key_layer = self.transformer.position_embeddings(
            query_layer, key_layer, kw_args['position_ids'],max_seqlen=kw_args['position_ids'].max()+1,
            layer_id=kw_args['layer_id']
        )

    _sp_text_length = None
    _sp_gp = None
    if "sp_pad" in kw_args:
        _sp_pad = kw_args["sp_pad"]
        # print(11111111111111111*4,kw_args["sp_pad"])#0
    else:
        _sp_pad = 0
    if "text_length_split" in kw_args and "sp_gp" in kw_args and kw_args["sp_gp"] != None:
        _sp_text_length = kw_args["text_length_split"]
        _sp_gp = kw_args["sp_gp"]
        q = all_to_all_with_pad(query_layer, _sp_gp, scatter_dim=1, gather_dim=2)
        k = all_to_all_with_pad(key_layer, _sp_gp, scatter_dim=1, gather_dim=2)
        v = all_to_all_with_pad(value_layer, _sp_gp, scatter_dim=1, gather_dim=2)
        query_layer = _remove_extra_encoder(q, _sp_text_length, _sp_gp.size(), pad=_sp_pad)
        key_layer = _remove_extra_encoder(k, _sp_text_length, _sp_gp.size(), pad=_sp_pad)
        value_layer = _remove_extra_encoder(v, _sp_text_length, _sp_gp.size(), pad=_sp_pad)

    # print("###############")
    # print(23333333333*2,"text_length_split" in kw_args,"sp_gp" in kw_args,kw_args["sp_gp"] != None,"sp_pad" in kw_args,query_layer.shape,key_layer.shape,value_layer.shape)#TTTT#([1, 12, 33886, 64])#[1, 6, 33890, 64]
    # print("###############")
    context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

    # print("###############")
    # print(444444444*2,_sp_text_length!=None,_sp_gp != None)#tt
    # print("###############")
    if _sp_text_length != None and _sp_gp != None:
        _hidden_states = _add_extra_encoder(context_layer, _sp_text_length, _sp_gp.size(), pad=_sp_pad)
        context_layer = all_to_all_with_pad(_hidden_states, _sp_gp, scatter_dim=2, gather_dim=1)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)

    if self.training:
        output = self.output_dropout(output)
    return output

def _add_extra_encoder(hidden_states, text_seq_length, sp_size, pad = 0):
    _dim = 2
    # add padding for split and later all2all
    # if pad is removed later than this
    # the split size will be wrong
    if pad > 0:
        # _pad_cfg_size = 2 * (len(hidden_states.shape) - _dim)
        # _pad_cfg = _pad_cfg_size * [0]
        # _pad_cfg[1] = pad
        # hidden_states = F.pad(hidden_states, _pad_cfg, mode="replicate")

        # _pad_tensor = torch.index_select(hidden_states, _dim, torch.tensor([-1], device=hidden_states.device))
        # _pad_shape = [1 for _ in _pad_tensor.shape]
        # _pad_shape[_dim] = pad
        # _pad_tensor = _pad_tensor.repeat(_pad_shape)
        # hidden_states = torch.cat([hidden_states, _pad_tensor], dim=_dim)

        # _pad_shape = list(hidden_states.shape)
        # _pad_shape[_dim] = pad
        # _pad_tensor = torch.zeros(_pad_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        # with open("/DATA/bvac/personal/opensora/zhipu/newsp/to_submit/same_sp_data_fix49addsppad/sat/shape.txt","a+")as f:
        #     f.write("%s\t%s\n"%(str(_pad_shape),str(hidden_states.shape)))
        #pad repeat
        _pad_tensor = hidden_states[:,:, -2:-1, :].repeat(1,1, pad, 1)
        hidden_states = torch.cat([hidden_states, _pad_tensor], dim=_dim)

    # current layout is [text, seq]
    # we want to add the extra encoder info [text, 1/n seq, text, 1/n seq, ...]
    encoder = hidden_states[:, :, :text_seq_length]
    seq = hidden_states[:, :, text_seq_length:]
    seq = seq.split(seq.size(_dim) // sp_size, dim=_dim)
    new_seq = []
    for i in range(sp_size):
        new_seq.append(encoder)
        new_seq.append(seq[i])

    hidden_states = torch.cat(new_seq, dim=_dim)
    return hidden_states

def _remove_extra_encoder(hidden_states, text_seq_length, sp_size, pad = 0):
    # current layout is [text, 1/n seq, text, 1/n seq, ...]
    # we want to remove the all the text info [text, seq]
    # print("###########")
    # print(2333333333, text_seq_length, sp_size,pad)#226#8#4
    # print("###########")
    _dim = 2
    split_seq = hidden_states.split(hidden_states.size(_dim) // sp_size, dim=_dim)
    encoder_hidden_states = split_seq[0][:, :, :text_seq_length]
    new_seq = [encoder_hidden_states]
    for i in range(sp_size):
        new_seq.append(split_seq[i][:, :, text_seq_length:])
    hidden_states = torch.cat(new_seq, dim=_dim)

    # remove padding added when all2all
    # if pad is removed earlier than this
    # the split size will be wrong
    if pad > 0:
        hidden_states = hidden_states.narrow(_dim, 0, hidden_states.size(2) - pad)
    return hidden_states

def cross_attention_forward_default(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].cross_attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_query_layer = self.query(hidden_states)
    query_layer = self._transpose_for_scores(mixed_query_layer)
    dropout_fn = self.attention_dropout if self.training else None
    if isinstance(encoder_outputs, torch.Tensor):
        mixed_x_layer = self.key_value(encoder_outputs)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)
        # Reshape and transpose [b, np, s, hn]
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        mem_cross = (key_layer, value_layer)
    else:
        key_layer, value_layer = encoder_outputs[kw_args['layer_id']]
        mem_cross = (key_layer, value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, cross_attention=True, mem_cross=mem_cross, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    # [b, s, hp]
    context_layer = context_layer.view(*new_context_layer_shape)

    # Output. [b, s, h]
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output

def routing_forward_default(self, hidden_states, **kw_args):
    num_experts = self.transformer.num_experts
    # This is just an example that select 2 experts randomly.
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = torch.randn((batch_size*sequence_length, num_experts), device=hidden_states.device, dtype=hidden_states.dtype)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)
    return routing_weights, selected_experts

from functools import partial

def mlp_forward_default(self, hidden_states, expert_id=-1, **kw_args):
    if self.transformer.num_experts == 1 or expert_id > -1:
        self = self.transformer.layers[kw_args['layer_id']].mlp
        suffix = f"_{expert_id}" if expert_id > 0 else ""
        if self.is_gated_mlp:
            intermediate_parallel = getattr(self, "dense_h_to_4h"+suffix)(hidden_states)
            gated_intermediate_parallel = getattr(self, "dense_h_to_4h_gate"+suffix)(hidden_states)
            intermediate_parallel = self.activation_func(gated_intermediate_parallel) * intermediate_parallel
            output = getattr(self, "dense_4h_to_h"+suffix)(intermediate_parallel)
        else:
            intermediate_parallel = getattr(self, "dense_h_to_4h"+suffix)(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            output = getattr(self, "dense_4h_to_h"+suffix)(intermediate_parallel)
        return output
    else:
        mlp_forward = self.hooks.get('mlp_forward', partial(mlp_forward_default, self))
        routing_forward = self.hooks.get('routing_forward', partial(routing_forward_default, self))
        self = self.transformer.layers[kw_args['layer_id']].mlp
        fwd_weight, fwd_idx = routing_forward(hidden_states, **kw_args)

        # Adapted from mixtral-8x7b https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(fwd_idx, num_classes=self.num_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[top_x_list] # I don't know why using hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = mlp_forward(current_state, expert_id=expert_idx, **kw_args) * fwd_weight[top_x_list, idx_list, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        output = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return output

def word_embedding_forward_default(self, input_ids, output_cross_layer, **kw_args):
    return self.transformer.word_embeddings(input_ids)

def position_embedding_forward_default(self, position_ids, output_cross_layer, **kw_args):
    if not self.transformer.is_rotary_emb:
        return self.transformer.position_embeddings(position_ids)
    return None
        
from sat.mpu import gather_from_model_parallel_region
def final_forward_default(self, logits, **kw_args):
    logits_parallel = F.linear(logits, self.transformer.word_embeddings.weight)
    if not kw_args['parallel_output']:
        logits_parallel = gather_from_model_parallel_region(logits_parallel)
    return logits_parallel

def layer_forward_default(self, hidden_states, mask, *args, **kw_args):
    '''
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
    '''
    self = self.transformer.layers[kw_args['layer_id']]
    
    # Layer norm at the begining of the transformer layer.
    attention_input = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output = self.attention(attention_input, mask, **kw_args)

    # Third LayerNorm
    if self.layernorm_order == 'sandwich':
        attention_output = self.third_layernorm(attention_output)

    # DropPath for attention
    if self.training and self.drop_path > 0.:
        # drop_path percentage 0, others 1/(1-p)
        random_tensor = (1-self.drop_path
                            + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1-self.drop_path)
        attention_output = random_tensor.view(-1, 1, 1) * attention_output
    
    # Residual connection.
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
    else:
        hidden_states = hidden_states + attention_output

    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            # Cross attention
            if self.layernorm_order == 'post':
                attention_output = self.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = mlp_input + attention_output
                # Layer norm post the cross attention
                mlp_input = self.post_cross_attention_layernorm(hidden_states)
            else:
                cross_input = self.post_cross_attention_layernorm(hidden_states)
                attention_output = self.cross_attention(cross_input, **kw_args)
                hidden_states = hidden_states + attention_output

    if self.layernorm_order != 'post':
        mlp_input = self.post_attention_layernorm(hidden_states)    

    # MLP.
    mlp_output = self.mlp(mlp_input, **kw_args)

    # Fourth LayerNorm
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)

    # DropPath for mlp
    if self.training and self.drop_path > 0.:
        random_tensor = (1-self.drop_path
                            + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1-self.drop_path)
        mlp_output = random_tensor.view(-1, 1, 1) * mlp_output

    # Second residual connection.
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output

    return output

HOOKS_DEFAULT = {
    'attention_fn': attention_fn_default,
    'attention_forward': attention_forward_default,
    'cross_attention_forward': cross_attention_forward_default,
    'routing_forward': routing_forward_default,
    'mlp_forward': mlp_forward_default,
    'word_embedding_forward': word_embedding_forward_default,
    'position_embedding_forward': position_embedding_forward_default,
    'final_forward': final_forward_default,
    'layer_forward': layer_forward_default
}

ARGS_DEFAULT = {
    'embedding_dropout_prob': ('hidden_dropout', 0),
    'attention_dropout_prob': ('attention_dropout', 0),
    'output_dropout_prob': ('hidden_dropout', 0),
    'inner_hidden_size': ('inner_hidden_size', None),
    'hidden_size_per_attention_head': ('hidden_size_per_attention_head', None),
    'cross_hidden_size_per_attention_head': ('cross_hidden_size_per_attention_head', None),
    'checkpoint_activations': ('checkpoint_activations', False),
    'checkpoint_num_layers': ('checkpoint_num_layers', 1),
    'checkpoint_skip_layers': ('checkpoint_skip_layers', 0),
    'is_decoder': ('is_decoder', False),
    'cross_attn_hidden_size': ('cross_attn_hidden_size', None),
    'use_final_layernorm': ('use_final_layernorm', True),
    'layernorm_epsilon': ('layernorm_epsilon', 1e-5),
    'use_bias': ('use_bias', True),
    'use_qkv_bias': ('use_qkv_bias', False),
    'num_multi_query_heads': ('num_multi_query_heads', 0),
    'cross_num_multi_query_heads': ('cross_num_multi_query_heads', 0),
    'drop_path': ('drop_path', 0.),
    'row_parallel_linear_final_bias': ('row_parallel_linear_final_bias', True),
    'is_gated_mlp': ('is_gated_mlp', False),
    'is_rotary_emb': ('is_rotary_emb', False),
    'parallel_output': ('parallel_output', False),
    'num_experts': ('num_experts', 1),
}

from sat.ops.layernorm import LayerNorm, RMSNorm

NO_WD_MODULES = [LayerNorm, torch.nn.LayerNorm, RMSNorm]