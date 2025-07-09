from sat.mpu import mp_split_model
from sat.model import BertModel
import torch
import os

def test_bert_inference():
    with torch.no_grad():
        model, args = BertModel.from_pretrained('bert-base-uncased')
        model = model.to(int(os.getenv('LOCAL_RANK')))
        model_device = next(model.parameters()).device
        x = torch.ones((8, 256), device=model_device, dtype=torch.long)
        inp = x.clone(); inp[:, -10:]*=30000 # split vocab
        a = model(input_ids=inp, position_ids=x, attention_mask=None, token_type_ids=x*0)
        mp_split_model(model, 2)

        b = model(input_ids=inp, position_ids=x, attention_mask=None, token_type_ids=x*0)

    print((a[0]-b[0]).abs().max())
    # assert torch.allclose(a[0], b[0])

if __name__ == '__main__':
    # torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/test_model_parallel.py
    print(os.getenv('RANK'))
    test_bert_inference()