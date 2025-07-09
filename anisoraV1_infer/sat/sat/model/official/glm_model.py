import torch
import torch.nn as nn

from sat.model.base_model import BaseModel, BaseMixin

class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings

class GLMModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.add_mixin('block_position_embedding', 
            BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size)
        )
    
    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for GLM"""
        group = parser.add_argument_group('GLM', 'GLM Configurations')
        group.add_argument('--block-lm', action='store_true', help="whether use the BlockLM pre-training")
        group.add_argument('--masked-lm', action='store_true', help='whether to use the mlm objective')
        group.add_argument('--bert-prob', type=float, default=0.5)
        group.add_argument('--gpt-infill-prob', type=float, default=0.5)
        group.add_argument('--gpt-min-ratio', type=float, default=0.5)
        group.add_argument('--gap-sentence-prob', type=float, default=0.0)
        group.add_argument('--gap-sentence-ratio', type=float, default=0.15)
        group.add_argument('--avg-block-length', type=int, default=3)
        group.add_argument('--short-seq-prob', type=float, default=0.0)
        group.add_argument('--single-span-prob', type=float, default=0.0)
        group.add_argument('--task-mask', action='store_true', help="Use different mask for generation and blank filling")
        group.add_argument('--no-shuffle-block', action='store_true', help="not shuffle the blocks when filling the blank")
        group.add_argument('--no-block-position', action='store_true',
                        help='Use (rough) absolute positions instead of block positions')
        group.add_argument('--sentinel-token', action='store_true',
                        help="Use sentinel (mask) tokens to replace 2d position encoding")
        group.add_argument('--block-mask-prob', type=float, default=0.0)
        group.add_argument('--context-mask-ratio', type=float, default=0.0)
        group.add_argument('--random-position', action='store_true',
                        help="Use random start position to cover all the position embeddings")
        group.add_argument('--cloze-eval', action='store_true', help='Evaluation dataset with cloze task')
        group.add_argument('--old-checkpoint', action='store_true', help="Loading the checkpoint from old libraray")

        group.add_argument('--tokenizer-model-type', type=str,
                       default=None,
                       help="Model type to use for sentencepiece tokenization \
                           (one of ['bpe', 'char', 'unigram', 'word']) or \
                           bert vocab to use for BertWordPieceTokenizer (one of \
                           ['bert-large-uncased', 'bert-large-cased', etc.])")
        return parser