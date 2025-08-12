import os
import torch
import argparse
from sat import get_args
from sat.model.official.distill_model import DistillModel
from bert_ft_model import ClassificationModel
args = get_args()

model, args = DistillModel.from_pretrained(ClassificationModel, args, 'checkpoints/finetune-bert-distill-boolq07-17-12-01', ClassificationModel)

model = model.student

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained(os.path.join('', 'bert-base-uncased'))

model.eval()
with torch.no_grad():
    text = [["This is a piece of text.", "Another piece of text."]]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    print(swiss_output)
 
# breakpoint()