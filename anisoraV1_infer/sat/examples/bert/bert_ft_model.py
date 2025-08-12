from sat.model.official.bert_model import BertModel
from sat.model.mixins import MLPHeadMixin

class ClassificationModel(BertModel):
    def __init__(self, args, transformer=None, layernorm_epsilon=1e-12, **kwargs):
        super().__init__(args, transformer=transformer, layernorm_epsilon=layernorm_epsilon, **kwargs)
        self.del_mixin('bert-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-finetune', 'BERT finetune Configurations')
        # group.add_argument('--prefix_len', type=int, default=16)
        return super().add_model_specific_args(parser)