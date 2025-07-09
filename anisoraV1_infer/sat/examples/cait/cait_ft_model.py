import torch
from sat.model.base_model import BaseMixin, non_conflict
from sat.model.official.vit_model import ClsMixin
from sat.model.mixins import PrefixTuningMixin
from sat.model.official.cait_model import CaiT

class CaiTFinetuneModel(CaiT):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon)
        self.decoder.del_mixin('cls')
        self.decoder.add_mixin('finetune_head', ClsMixin(args.hidden_size, args.num_finetune_classes))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CaiT-finetune', 'CaiT finetuning Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        return super().add_model_specific_args(parser)