from sat.model.official.bert_model import BertModel

class RobertaModel(BertModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(RobertaModel, self).__init__(args, transformer=transformer, **kwargs)
        self.del_mixin("bert-type")
