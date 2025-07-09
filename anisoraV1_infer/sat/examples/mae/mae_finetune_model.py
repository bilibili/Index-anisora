import torch

class MAE_finetune(torch.nn.Module):
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.final = torch.nn.Linear(hidden_size, num_classes)
        self.encoder = encoder
    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        x, *mem = self.encoder(tokens, position_ids, attention_mask, **kwargs)
        x = self.final(x[:, 0])
        return x
    def disable_untrainable_params(self):
        self.encoder.transformer.position_embeddings.requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('MAE-finetune', 'MAE finetuning Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        return parser
