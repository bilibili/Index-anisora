from xformers.ops import memory_efficient_attention as mea
from xformers.ops import LowerTriangularMask

def memory_efficient_attention(q, k, v, attention_dropout=0., mask=None, scale=None):
    if mask is None:
        attn_bias = None
    else:
        t = (mask > 0)
        if t.all():
            attn_bias = None
        elif not t.triu(diagonal=1).any() and t.tril().all():
            attn_bias = LowerTriangularMask()
        else:
            raise ValueError(f"Unknown mask type {mask}")
    return mea(q, k, v, attn_bias=attn_bias, p=attention_dropout, scale=scale)
