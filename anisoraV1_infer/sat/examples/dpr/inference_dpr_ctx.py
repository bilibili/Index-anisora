import os
import torch
from sat import get_args, AutoModel

args = get_args()

model_type = 'dpr-ctx_encoder-single-nq-base'
print(model_type)
model, args = AutoModel.from_pretrained(model_type, args)

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
dpr = DPRContextEncoder.from_pretrained(os.path.join('', model_type), output_hidden_states=True)
tokenizer = DPRContextEncoderTokenizer.from_pretrained(os.path.join('', model_type))

model.eval()
dpr.eval()
with torch.no_grad():
    text = "Hello, is my dog cute ?"
    encoded_input = tokenizer(text, return_tensors='pt')
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    hugging_output = dpr(**encoded_input)[0]
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    # swiss_output = swiss_output[:, 0, :]
    print(position_ids)
    print("hugging_output = ", hugging_output)
    print("swiss_output = ", swiss_output)
    # Since we don't use padding_idx for Embedding layers, pad output is largely different between hugging and swiss.
    # You will find it if you calculate error for hugging_output[1] and swiss_output[1].
    # However, pad output is usually not used, it doesn't matter too much.
    print("max error:", (hugging_output[:,0] - swiss_output[:,0]).abs().max())
    print("max relative error:", ((hugging_output[:,0] - swiss_output[:,0]).abs() / torch.max(swiss_output[:,0].abs(), hugging_output[:,0].abs())).max())
