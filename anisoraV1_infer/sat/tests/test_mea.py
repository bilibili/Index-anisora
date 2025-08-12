from sat.ops import memory_efficient_attention
import torch

def test_mea(seq_len, hidden_size, num_heads, batch_size=8, qkv=None):
    torch.cuda.manual_seed(0)
    q, k, v = qkv.clone().cuda().requires_grad_(True)
    # profile time and memory
    torch.cuda.reset_peak_memory_stats()
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        out = memory_efficient_attention(q, k, v)
        out.sum().backward()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
    print(torch.cuda.max_memory_allocated() / 1024 ** 3)
    # clean cuda cache
    del q, k, v
    torch.cuda.empty_cache()
    return out

def attention(query, key, value):
        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = torch.nn.functional.dropout(attn, 0)
        return attn @ value

def test_standard_attention(seq_len, hidden_size, num_heads, batch_size=8, qkv=None):
    # set random seed
    q, k, v = qkv.clone().cuda().transpose(2,3).requires_grad_(True)
    # clean memory record
    torch.cuda.reset_peak_memory_stats()
    # profile time and memory
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        out = attention(q, k, v)
        out.sum().backward()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # peak memory usage
    print(torch.cuda.max_memory_allocated() / 1024 ** 3)
    
    # clean cuda cache
    del q, k, v
    torch.cuda.empty_cache()
    return out.transpose(1,2)

def test_mixin():
    with torch.no_grad():
        from sat.model import BaseModel, AutoModel
        from sat.model.mixins import MemoryEfficientAttentionMixin, TransposedMemoryEfficientAttentionMixin
        model = BaseModel(args=BaseModel.get_args(
             max_sequence_length=5000,
        ))
        model = model.cuda().eval().half()
        x = torch.tensor([range(4096)], device='cuda')
        with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
            a = model(input_ids=x, position_ids=x, attention_mask=None)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        model.add_mixin('mea', TransposedMemoryEfficientAttentionMixin())
        with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
            b = model(input_ids=x, position_ids=x, attention_mask=None)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print(((a[0]-b[0]).abs()/(a[0].abs().mean())).max())

if __name__ == "__main__":
    # seq_len = 2048
    # hidden_size = 640 // 16
    # num_heads = 16
    # batch_size = 8
    # qkv = torch.randn(3, batch_size, seq_len, num_heads, hidden_size).half()
    # out_mea = test_mea(seq_len, hidden_size, num_heads, batch_size, qkv)
    # out_std = test_standard_attention(seq_len, hidden_size, num_heads, batch_size, qkv)
    # print((out_mea-out_std).abs().max())

    test_mixin()