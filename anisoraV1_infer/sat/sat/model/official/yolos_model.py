import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model import ViTModel
from sat.model.official.vit_model import ImagePatchEmbeddingMixin
from sat.model.mixins import BaseMixin


class NewTokenMixin(ImagePatchEmbeddingMixin):
    def __init__(self, new_token_size, in_channels, hidden_size, property, init_method_std=0.02):
        super(NewTokenMixin, self).__init__(in_channels, hidden_size, property)
        self.new_token_size = new_token_size
        self.init_method_std = init_method_std
    
    def reinit(self, parent_model=None):
        old_weight = self.transformer.word_embeddings.weight.data
        self.transformer.word_embeddings = torch.nn.Embedding(self.new_token_size, old_weight.shape[1]).type(old_weight.dtype).to(old_weight.device)
        torch.nn.init.normal_(self.transformer.word_embeddings.weight, mean=0.0, std=self.init_method_std)
        self.transformer.word_embeddings.weight.data[:old_weight.shape[0]] = old_weight

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DetHeadMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.num_det_tokens = args.num_det_tokens
        self.class_embed = MLP(args.hidden_size, args.hidden_size, args.num_det_classes, 3)
        self.bbox_embed = MLP(args.hidden_size, args.hidden_size, 4, 3)
    
    def final_forward(self, logits, **kw_args):
        logits = logits[:, -self.num_det_tokens:]
        outputs_class = self.class_embed(logits)
        outputs_coord = self.bbox_embed(logits).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out

class YOLOS(ViTModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-6, **kwargs):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, **kwargs)
        self.del_mixin('patch_embedding')
        self.add_mixin('patch_embedding', NewTokenMixin(args.vocab_size+args.num_det_tokens, args.in_channels, args.hidden_size, self.property))
        self.del_mixin('cls')
        self.add_mixin('det_head', DetHeadMixin(args))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('YOLOS', 'YOLOS Configurations')
        group.add_argument('--num-det-tokens', type=int)
        group.add_argument('--num-det-classes', type=int)
        return super().add_model_specific_args(parser)