import os
import torch
import argparse
from sat import get_args, AutoModel

args = get_args()

model_type = 'eva02_L_pt_m38m_p14'
model, args = AutoModel.from_pretrained(model_type, args, layernorm_epsilon=1e-6)

x = torch.randn(2, 3, 224, 224).half().cuda()
bool_mask = torch.ones(2, 256, dtype=torch.bool).cuda()
bool_mask[:, 1] = False
input_ids = torch.zeros(2, 1, dtype=torch.long).cuda()
attention_mask = torch.tensor([[1.]], dtype=torch.float16).cuda()


model.eval().cuda()
with torch.no_grad():
    dst_output = model(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=x, bool_masked_pos=bool_mask)
    print(dst_output[0])
    # The output is slightly different from original transformed eva2 model. It is because that rope params become half due to --fp16 argument.