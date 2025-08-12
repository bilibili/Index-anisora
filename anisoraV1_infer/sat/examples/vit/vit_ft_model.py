import torch
from sat.model.base_model import BaseMixin, non_conflict
from sat.model.official.vit_model import ViTModel, ClsMixin
from sat.model.mixins import PrefixTuningMixin

class ViTFinetuneModel(ViTModel):
    def __init__(self, args, transformer=None, **kw_args):
        super().__init__(args, transformer=transformer, **kw_args)
        self.del_mixin('cls')
        self.add_mixin('finetune_head', ClsMixin(args.hidden_size, args.num_finetune_classes))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT-finetune', 'ViT finetuning Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        group.add_argument('--finetune-resolution', nargs='+', type=int, default=[384, 384])
        return super().add_model_specific_args(parser)