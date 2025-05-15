import torch
import torch_npu
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np

# Adapted from https://github.com/Dao-AILab/flash-attention/blob/0dfb28174333d9eefb7c1dd4292690a8458d1e89/flash_attn/flash_attn_interface.py#L522
def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q, k, v = torch.unbind(qkv, dim=1)
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

    head_num = q.size(1)
    if causal:
        atten_mask_npu = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).bool().to(device=qkv.device)
        out_padded = torch_npu.npu_fusion_attention(
                        q, k, v, head_num, \
                        pse=None, \
                        atten_mask=atten_mask_npu, \
                        scale=softmax_scale, \
                        keep_prob=1, \
                        input_layout="TND", \
                        actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()), \
                        actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()), \
                        sparse_mode=3)[0]
    else:
        out_padded = torch_npu.npu_fusion_attention(
                        q, k, v, head_num, \
                        pse=None, \
                        atten_mask=None, \
                        scale=softmax_scale, \
                        keep_prob=1, \
                        input_layout="TND", \
                        actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()), \
                        actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()))[0]

    out = out_padded[..., :head_size_og]
    return out

def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False
):
    if causal == False:
        head_num = q.shape[1]
        output = torch_npu.npu_fusion_attention(q, k, v, head_num,
                                                pse=None,
                                                atten_mask=None,
                                                scale=1.0 / math.sqrt(q.shape[-1]),
                                                keep_prob=1,
                                                input_layout="TND",
                                                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()))[0]
    else:
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)
        head_num = q.shape[1]
        output = torch_npu.npu_fusion_attention(q, k, v, head_num,
                                                pse=None,
                                                padding_mask=None,
                                                atten_mask=atten_mask_npu,
                                                scale=1.0 / math.sqrt(q.shape[-1]),
                                                keep_prob=1,
                                                input_layout="TND",
                                                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                                                sparse_mode=3)[0]
    return output
