import torch
import torch.nn as nn
from sat.model.base_model import BaseMixin, BaseModel

class DPREncoderFinalMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def final_forward(self, logits, **kwargs):
        logits = logits[:, 0, :]
        return logits

class DPRReaderFinalMixin(BaseMixin):
    def __init__(self, hidden_size, projection_dim):
        super().__init__()
        if projection_dim > 0:
            embeddings_size = projection_dim
        else:
            embeddings_size = hidden_size
        self.qa_outputs = nn.Linear(embeddings_size, 2)
        self.qa_classifier = nn.Linear(embeddings_size, 1)

    def final_forward(self, logits, **kwargs):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        print("Before final_forward: logits = ", logits)
        n_passages, sequence_length = logits.size()[:2]
        sequence_output = logits

        # compute logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])

        # resize
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)

        return (start_logits, end_logits, relevance_logits)

class DPRTypeMixin(BaseMixin):
    def __init__(self, num_types, hidden_size):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, hidden_size)
        
    def word_embedding_forward(self, input_ids, **kwargs):
        print("DPRTypeMixin word_embedding_forward")
        if "token_type_ids" in kwargs:
            token_type_ids = kwargs["token_type_ids"]
        else:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long).to(input_ids.device)
        return self.transformer.word_embeddings(input_ids) + self.type_embeddings(token_type_ids)

class DPRQuestionEncoder(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(DPRQuestionEncoder, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin("dpr-type", DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin("dpr-final", DPREncoderFinalMixin())
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRQuestionEncoder', 'DPRQuestionEncoder Configurations')
        group.add_argument('--num-types', type=int)
        return parser

class DPRContextEncoder(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(DPRContextEncoder, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin("dpr-type", DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin("dpr-final", DPREncoderFinalMixin())
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRContextEncoder', 'DPRContextEncoder Configurations')
        group.add_argument('--num-types', type=int)
        return parser

class DPRReader(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(DPRReader, self).__init__(args, transformer=transformer, **kwargs)
        self.add_mixin("dpr-type", DPRTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin("dpr-final", DPRReaderFinalMixin(args.hidden_size, args.projection_dim))
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('DPRReader', 'DPRReader Configurations')
        group.add_argument('--num-types', type=int)
        group.add_argument('--projection-dim', type=int)
        return parser
    