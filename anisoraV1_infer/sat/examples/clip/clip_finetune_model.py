import torch
from sat.model.official.clip_model import CLIP

class CLIP_finetune(torch.nn.Module):
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.final = torch.nn.Linear(hidden_size, num_classes)
        self.encoder = encoder
    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        x, *mem = self.encoder(tokens, position_ids, attention_mask, **kwargs)
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.final(x)
        return x
    def disable_untrainable_params(self):
        self.encoder.transformer.position_embeddings.requires_grad_(False)
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CLIP-ft', 'CLIP-ft')
        group.add_argument("--num-finetune-classes", type=int, default=None)
        return parser

class CLIP_wp(CLIP):
    def disable_untrainable_params(self):
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        self.logit_scale.requires_grad_(False)
        self.image_encoder.transformer.position_embeddings.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CLIP-ft', 'CLIP-ft')
        group.add_argument("--num-finetune-classes", type=int, default=None)
        return super().add_model_specific_args(parser)