import torch.nn as nn

class DistillModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def forward(self, teacher_kwargs, student_kwargs):
        teacher_logits, *mem_t = self.teacher(**teacher_kwargs)
        student_logits, *mem_s = self.student(**student_kwargs)
        return teacher_logits, student_logits
    
    def disable_untrainable_params(self):
        for n, p in self.teacher.named_parameters():
            p.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-distill', 'BERT distill Configurations')
        group.add_argument('--teacher', type=str)
        group.add_argument('--tc-type', type=str)
        group.add_argument('--st-type', type=str)
        return parser
    
    @classmethod
    def from_pretrained(cls, args, teacher_cls, student_name, student_cls):
        student, args = student_cls.from_pretrained(student_name, args, prefix='student.')
        if isinstance(teacher_cls, type):
            teacher, t_args = teacher_cls.from_pretrained(args.teacher, args)
        else:
            teacher = teacher_cls
        model = DistillModel(teacher, student)
        return model, args