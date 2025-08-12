import torch
import argparse
import numpy as np

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
import torch.nn as nn
from bert_ft_model import ClassificationModel

from sat.model.official import DistillModel

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'token_type_ids', 'attention_mask', 'label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['label'].long()
    position_ids = data_b['position_ids'].long()
    token_type_ids = data_b['token_type_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    
    return tokens, labels, attention_mask, position_ids, token_type_ids, (tokens!=1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, token_type_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    teacher_kwargs = dict(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    student_kwargs = dict(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

    teacher_logits, student_logits = model(teacher_kwargs, student_kwargs)
    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = student_logits.contiguous().float().squeeze(-1)[..., 0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    loss_distill = nn.MSELoss()(teacher_logits, student_logits)
    total_loss = loss + 0.1 * loss_distill
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return total_loss, {'acc': acc}

def _encode(text, text_pair):
    tokenizer = get_tokenizer()
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    seq_len = len(encoded_input['input_ids'])
    position_ids = torch.arange(seq_len)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids.numpy(), token_type_ids=encoded_input['token_type_ids'], attention_mask=encoded_input['attention_mask'])

from sat.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack, label = _encode(row['passage'], row['question']), int(row['label'])
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'token_type_ids': np.array(pack['token_type_ids'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "token_type_ids", "attention_mask", "label"], cache_dir=args.data_root, offline=False, transformer_name="boolq_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--data_root', type=str)
    py_parser = DistillModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    student_model, student_args = ClassificationModel.from_pretrained(args.st_type, args)
    teacher_model, teacher_args = ClassificationModel.from_pretrained(args.teacher, args)
    model = DistillModel(teacher_model, student_model)
    
    get_tokenizer(student_args)
    training_main(student_args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
