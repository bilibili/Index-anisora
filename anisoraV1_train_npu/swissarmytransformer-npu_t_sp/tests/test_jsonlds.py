from sat.data_utils import make_loaders
from sat import get_args
from sat.data_utils import JsonlIterableDataset

from transformers import AutoTokenizer
from functools import partial
import torch

args = get_args(['--batch-size', '2', 
'--train-data', 
'/mnt/shared/img_datasets/text_dm/code/python.jsonl,/mnt/shared/txt_datasets/book', 
'/mnt/shared/img_datasets/text_dm/wudao/0.jsonl', '--train-data-weights', '2', '1', 
'--iterable-dataset', '--split','1', '--num-workers', '0'])

def process_fn(src, seq_len):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/shared/official_pretrains/hf_home/chatglm2-6b", trust_remote_code=True, local_files_only=True)

    buffered_token_ids = None
    eos_id = torch.tensor([tokenizer.eos_token_id])

    for x in src:
        type_id = 0 if 'code' in x else 1
        txt = x['code'] if 'code' in x else x['content']
        tokenized_ids = torch.tensor(tokenizer.encode(txt))
        if buffered_token_ids is None:
            buffered_token_ids = tokenized_ids
        else:
            buffered_token_ids = torch.cat((buffered_token_ids, eos_id, tokenized_ids), dim=0)
        # yield per seq_len
        while buffered_token_ids.shape[0] >= seq_len:
            yield {'txt': buffered_token_ids[:seq_len], 'type_id': type_id}
            buffered_token_ids = buffered_token_ids[seq_len:]
        
def create_func(path, args):
    return JsonlIterableDataset(path, partial(process_fn, seq_len=512), seed=1, shuffle_buffer=1)

train_loader, val_loader, test_loader = make_loaders(args,create_dataset_function=create_func)

tokenizer = AutoTokenizer.from_pretrained("/mnt/shared/official_pretrains/hf_home/chatglm2-6b", trust_remote_code=True, local_files_only=True)
a, b = 0, 0
for i, x in enumerate(train_loader):
    # print(tokenizer.decode(x['txt'][0]))
    # print('-------------------')
    if x['type_id'][0].item() == 0:
        a += 1
    else: 
        b += 1
    if i > 100:
        break

print(a, b)



