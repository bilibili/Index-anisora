from sat.model.base_model import BaseModel
import torch

def test_list_avail_args():
    a = BaseModel.list_avail_args()
    assert a.parse_args([]).num_layers == 24
    from sat.model import GLMModel
    a = GLMModel.list_avail_args()
    assert hasattr(a.parse_args([]), 'gpt_infill_prob')
    
def test_model_get_args():
    from sat.model import GLMModel
    args = GLMModel.get_args()
    assert args.num_layers == 24
    print(args)
    args = GLMModel.get_args(num_layers=2)
    assert args.num_layers == 2

def test_model_from_pretrained():
    from sat.model import RobertaModel, AutoModel
    model, args1 = RobertaModel.from_pretrained('roberta-base')
    print(args1)

    model, args2 = AutoModel.from_pretrained('roberta-base')
    # compare args one by one 
    for k, v in args1.__dict__.items():
        assert getattr(args2, k) == v
        
def test_auto_init_model_only():
    # check not torch.distributed.is_initialized()
    assert not torch.distributed.is_initialized()
    from sat.model import AutoModel
    model, args = AutoModel.from_pretrained('roberta-base')
    print(args)

    
    
if __name__ == '__main__':
    from argparse import Namespace, ArgumentParser
    from sat.arguments import _simple_init, get_args
    # args_full = get_args([])
    args = Namespace(
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
    )
    # md = BaseModel(args)
    # _simple_init(args.model_parallel_size)
    test_list_avail_args()
    # test_model_get_args()
    # test_model_from_pretrained()
    # test_auto_init_model_only()
