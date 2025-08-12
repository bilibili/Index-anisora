from sat.model import BaseModel, BertModel, BaseMixin
import torch
import json, os

class NestedModel(BaseModel):
    def __init__(self, args, *argv, **kwargs):
        super().__init__(args, *argv, **kwargs)
        self.net = BertModel(BertModel.get_args(
            **args.bert_args
        ))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Nested Example', 'Nested Example Configurations')
        group.add_argument('--bert_args', type=json.loads, default={}, help='Bert Model Arguments')
        return parser

    @BaseMixin.non_conflict
    def word_embedding_forward_default(self, input_ids, output_cross_layer, old_impl, **kw_args):
        return old_impl(input_ids, **kw_args) + self.net(input_ids, input_ids,input_ids)

def test_create():
    a = NestedModel(args=NestedModel.get_args(num_layers=2,
            bert_args={"hidden_size": 128, "num_layers": 3, 'num_types': 2} 
        ))
    assert a.net.transformer.position_embeddings.weight.shape[-1] == 128

def test_save_and_load():
    args = NestedModel.get_args(
        num_layers=2,
            bert_args={"hidden_size": 128, "num_layers": 3, 'num_types': 2} 
        )
    args.mode = 'inference'
    args.save = './checkpoints/test_nested_model'
    args.tokenizer_type = 'fake'
    a = NestedModel(args=args).cuda()

    from sat.training.model_io import save_checkpoint

    save_checkpoint(1, a, None, None, args)

    assert os.path.exists(
    os.path.join(args.save, str(1), 'mp_rank_00_model_states.pt'))

    b, args = NestedModel.from_pretrained(args.save)

    assert b.net.transformer.position_embeddings.weight.shape[-1] == 128

    # compare the weights equal between a and b
    for a_p, b_p in zip(a.parameters(), b.parameters()):
        assert torch.allclose(a_p, b_p)
    
def test_load():
    trained_dir = './checkpoints/test_train_nested/MyModel-05-11-01-35'

    b, args = NestedModel.from_pretrained(trained_dir)
    assert b.net.transformer.position_embeddings.weight.shape[-1] == 128


if __name__ == '__main__':
    test_create()
    test_save_and_load()
    # test_load()
