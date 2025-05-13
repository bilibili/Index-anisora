from sat.model.base_model import BaseModel
from sat import get_args
import torch
import argparse


def test_model_inference():
    from sat.model import RobertaModel, AutoModel
    model, args1 = RobertaModel.from_pretrained('roberta-base')
    x = torch.tensor([[1,2,3]], device='cuda')
    a = model(input_ids=x, position_ids=x, attention_mask=None)

def test_model_inference_create():
    from sat.model import RobertaModel, AutoModel
    model = RobertaModel(args = argparse.Namespace(
        num_layers=2,
        vocab_size=100,
        hidden_size=100,
        num_attention_heads=2,
        max_sequence_length=100,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        output_dropout_prob=0.1,
        inner_hidden_size=100,
        hidden_size_per_attention_head=50,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        layernorm_order='pre',
        skip_init=False,
        use_gpu_initialization=False,
        model_parallel_size=1,
        num_types=2,
    )
    ).cuda()
    x = torch.tensor([[1,2,3]], device='cuda')
    a = model(input_ids=x, position_ids=x, attention_mask=None)

def test_full_mode_inference():
    args = get_args(['--zero-stage', '1'])
    from sat.model import RobertaModel, AutoModel
    model, args1 = RobertaModel.from_pretrained('roberta-base')
    x = torch.tensor([[1,2,3]], device='cuda')
    a = model(input_ids=x, position_ids=x, attention_mask=None)

if __name__ == '__main__':
    # test_model_inference()
    test_model_inference_create()
