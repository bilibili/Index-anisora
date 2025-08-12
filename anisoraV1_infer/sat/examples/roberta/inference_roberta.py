import os
import torch
import argparse
from sat import get_args, AutoModel
# from sat.model.official.bert_model import BertModel

args = get_args()

model_type = 'roberta-base'
model, args = AutoModel.from_pretrained(model_type, args)

pretrain_path = ''
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from transformers import RobertaTokenizer, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, model_type))
roberta = RobertaForMaskedLM.from_pretrained(os.path.join(pretrain_path, model_type), output_hidden_states=True)

model.eval()
roberta.eval()
with torch.no_grad():
    text = ["This is a piece of text.", "Another piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    hugging_output = roberta(**encoded_input)[0]
    position_ids = create_position_ids_from_input_ids(encoded_input['input_ids'], 1, 0)
    print(position_ids)
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    # Since we don't use padding_idx for Embedding layers, pad output is largely different between hugging and swiss.
    # You will find it if you calculate error for hugging_output[1] and swiss_output[1].
    # However, pad output is usually not used, it doesn't matter too much.
    print("max error:", (hugging_output[0] - swiss_output[0]).abs().max())
    print("max relative error:", ((hugging_output[0] - swiss_output[0]).abs() / torch.max(swiss_output[0].abs(), hugging_output[0].abs())).max())

# breakpoint()