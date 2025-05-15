import math
import os
import torch
import torch.nn.functional as F
from einops import rearrange
import random

from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding
from sat.model.position_embedding.rotary_embeddings_original import RotaryEmbedding, apply_rotary_pos_emb_bhs


import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.random.manual_seed(0)

def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x.ndim-1)  # dim=-1 triggers a bug in earlier torch versions


class SatRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, device=torch.device('cpu')):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        # inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            cos_cached = cos_cached.to(x.dtype)
            sin_cached = sin_cached.to(x.dtype)
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


if __name__ == "__main__":
    headdim = 128
    batch_size = 10
    seqlen = 1000
    nheads = 64
    rotary_dim = headdim
    device = torch.device('cuda')
    dtype = torch.bfloat16
    max_seqlen = seqlen - 100
    position_ids = torch.arange(seqlen, device=device)
    position_ids = position_ids.repeat(batch_size, 1)
    
    for i in range(batch_size):
        for j in range(seqlen):
            position_ids[i, j] = random.randint(1, 800)
    
    position_ids.to(device)
    #print(position_ids)
    
    
    
    
    # test llama 
    query = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype, requires_grad=True)
    
    query_l = query.detach().clone().requires_grad_()
    key_l = key.detach().clone().requires_grad_()
    
    
    llama_rotary = SatRotaryEmbedding(
            rotary_dim,
            base=10000,
            precision=torch.bfloat16,
            learnable=False,
            device=device
    )
    
    cos, sin = llama_rotary(query, seq_len=max_seqlen)
    q, k = apply_rotary_pos_emb_index_bhs(query, key, cos, sin, position_ids)
    
    rotary_emb_llama = FastRotaryEmbedding(dim=rotary_dim, device=device)
    q_l, k_l = rotary_emb_llama(query_l, key_l, position_ids, max_seqlen=max_seqlen)

    g = torch.randn_like(q_l)
    g_og = g.clone().detach()  # If inplace=True, we might modify the gradient inplace
    

    print("llama")
    print("q max diff ", (q_l - q).abs().max().item())
    print("q mean diff ", (q_l - q).abs().mean().item())
    print("k max diff ", (k_l - k).abs().max().item())
    print("k mean diff ", (k_l - k).abs().mean().item())
    
    
    q_l.backward(g)
    #k_l.backward(g)
    q.backward(g_og)
    #k.backward(g_og)
    #print(query_l.grad)
    #print(query.grad)
    print("grad max diff ", (query_l.grad - query.grad).abs().max().item())
    print("grad mean diff ", (query_l.grad - query.grad).abs().mean().item())
    #print("grad max ", (key_l.grad - key.grad).abs().max().item())
    #print("grad mean ", (key_l.grad - key.grad).abs().mean().item())
    
    print("-------------------------------------------------")
    
    # test chatglm 
    query = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype, requires_grad=True)
    
    query_c = query.detach().clone().requires_grad_()
    key_c = key.detach().clone().requires_grad_()
    
    
    chatglm_rotary = RotaryEmbedding(headdim // 2, original_impl=True, device=device)
    rotary_pos_emb = chatglm_rotary(max_seqlen)
    rotary_pos_emb = rotary_pos_emb.to(torch.bfloat16)
    
    rotary_pos = rotary_pos_emb[position_ids]

    q = apply_rotary_pos_emb_bhs(query, rotary_pos)
    k = apply_rotary_pos_emb_bhs(key, rotary_pos)
    
    rotary_emb_chatglm = FastRotaryEmbedding(dim=headdim // 2, interleaved=True, device=device)
    q_c, k_c = rotary_emb_chatglm(query_c, key_c, position_ids, max_seqlen=max_seqlen)
    
    print("chatglm")
    print("q max diff ", (q_c - q).abs().max().item())
    print("q mean diff ", (q_c - q).abs().mean().item())
    print("k max diff ", (k_c - k).abs().max().item()) 
    print("k mean diff ", (k_c - k).abs().mean().item())
    
    g = torch.randn_like(q_c)
    g_og = g.clone().detach()  # If inplace=True, we might modify the gradient inplace
    q_c.backward(g)
    #k_c.backward(g)
    q.backward(g_og[:, :, :, :rotary_dim])
    #k.backward(g_og[:, :, :, :rotary_dim])
    print("q grad max diff ", (query_c.grad - query.grad).abs().max().item())
    print("q grad mean diff ", (query_c.grad - query.grad).abs().mean().item())
    #print("grad max ", (key_c.grad - key.grad).abs().max().item())
    #print("grad mean ", (key_c.grad - key.grad).abs().mean().item())
    
    
    print("-----------------------------------------------")
    print("性能测试")
    
    
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        q_triton,k_triton = rotary_emb_llama(query_l, key_l, position_ids, max_seqlen=seqlen)
    torch.cuda.synchronize()
    tt = time.time() - st
    print("now time is ", tt)
    
    
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        cos, sin = llama_rotary(query, seq_len=max_seqlen)
        qu, ke = apply_rotary_pos_emb_index_bhs(query, key, cos, sin, position_ids)
    torch.cuda.synchronize()
    l_tt = time.time() - st
    print("ref time is ", l_tt)
    
    print("llama speed up ", l_tt / tt)
    
    
    
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        q_triton,k_triton = rotary_emb_chatglm(query_c, key_c, position_ids, max_seqlen=seqlen)
    torch.cuda.synchronize()
    tt = time.time() - st
    print("now time is ", tt)
    has_nan_q = torch.isnan(q_triton).any()
    has_nan_k = torch.isnan(k_triton).any()
    if has_nan_q or has_nan_k:
        print("nan because of inplace, set inplace false")

    
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        rotary_pos = rotary_pos_emb[position_ids]
        q = apply_rotary_pos_emb_bhs(query, rotary_pos)
        k = apply_rotary_pos_emb_bhs(key, rotary_pos)
        
    torch.cuda.synchronize()
    c_tt = time.time() - st
    print("ref time is ", c_tt)
    
    print("chatglm speed up ", c_tt / tt)