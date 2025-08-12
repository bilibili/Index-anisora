import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
import math
from sat import mpu
from transformers.activations import ACT2FN

gelu = ACT2FN["gelu_new"]

class GPT2FinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

class GPT2Model(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(GPT2Model, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin("gpt2-final", GPT2FinalMixin(args.vocab_size, args.hidden_size))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('GPT2', 'GPT2 Configurations')
        # group.add_argument('--num-types', type=int)
        return parser