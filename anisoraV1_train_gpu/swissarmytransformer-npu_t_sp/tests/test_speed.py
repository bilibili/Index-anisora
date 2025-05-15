from sat.model import BaseModel
import torch

def test_speed():
    import time
    model = BaseModel(args=BaseModel.get_args(
        num_layers=40,
        hidden_size=2048,
        num_attn_heads=32,
        max_sequence_length=1024,
        vocab_size=32000,
        layernorm_order='sandwich',
        fp16=True
    )).cuda()
    data = torch.zeros(4, 1024, dtype=torch.long).cuda()
    start = time.time()
    for i in range(100):
        with torch.no_grad():
            model(data, data, None)
    print('average time: ', (time.time() - start) / 100, 's')

if __name__ == '__main__':
    test_speed()

