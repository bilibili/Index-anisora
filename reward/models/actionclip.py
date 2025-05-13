from typing import Any, Dict, List, Optional, Tuple, Union
import sys
sys.path.append('/workspace/CLIP-main')
import clip
import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel
from mmengine.structures import LabelData

from mmaction.registry import MODELS
from .adapter import TransformerAdapter


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


def text_prompt(labels_or_label_file, templates_or_template_file=None, mode='official'):
    if isinstance(labels_or_label_file, str):
        labels = mmengine.list_from_file(labels_or_label_file)
    elif isinstance(labels_or_label_file, list):
        labels = labels_or_label_file
    else:
        raise ValueError(f'`labels_or_label_file` must be `list` or `str`, '
                         f'but got {type(labels_or_label_file)}')

    if templates_or_template_file is None and mode == 'official':
        templates = [
            'a photo of action {}', 'a picture of action {}',
            'Human action of {}', '{}, an action', '{} this is an action',
            '{}, a video of action', 'Playing action of {}', '{}',
            'Playing a kind of action, {}', 'Doing a kind of action, {}',
            'Look, the human is {}', 'Can you recognize the action of {}?',
            'Video classification of {}', 'A video of {}', 'The man is {}',
            'The woman is {}'
        ]
        num_prompt = len(templates)
    elif templates_or_template_file is None and mode == 'self':
        templates = ['{}']
        num_prompt = len(templates)
    elif isinstance(templates_or_template_file, str):
        templates = mmengine.list_from_file(templates_or_template_file)
    elif not mmengine.is_seq_of(templates_or_template_file, str):
        raise ValueError(f'`template` must be list of `str`, `str` or `None`, '
                         f'but got {type(templates_or_template_file)}')

    # for c in labels:
    #     for t in templates:
    #         print(t.format(c))
    if mode == 'official':
        prompt = torch.cat(
            [clip.tokenize(t.format(c)) for t in templates for c in labels])
    elif mode == 'self':
        prompt_mapping = {
            # "The video contains multiple distinct shots, with significant changes in scenes or perspectives.": ['The video contains jump cuts, with significant changes in scenes or perspectives.', 'The protagonist is in a jump cutting scene'],
            # 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.': ['The video contains some person who has a large range of movement, such as running, jumping, dancing, or waving arms.', 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.'],
            # 'Camera movement, the protagonist has small-scale body movements': ['The video contains camera movement and the protagonist has small-scale body movements', 'The protagonist has small-scale body movements while the camera moves imparently.'],
            # 'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.': ['The video contains some people\'s detailed face with movements such as expressions, eye movements or mouth movements', 'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.'],
            # 'The protagonist remain stationary in the video with no apparent movement.': ['The video contains some people who remain stationary.', 'The protagonist remain stationary in the video with no apparent movement.'],
            # 'organism': ['The video contains no person.', 'The protagonist can not be found in the video.']

            # "The video contains multiple distinct shots, with significant changes in scenes or perspectives.": "The video contains jump cuts, with significant changes in scenes or perspectives.",
            # 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.': 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.',
            # 'Camera movement, the protagonist has small-scale body movements': 'The video contains camera movement. The protagonist has small-scale body movements',
            # 'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.': 'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.',
            # 'The protagonist remain stationary in the video with no apparent movement.': 'The protagonist remain stationary in the video with no apparent movement.',
            # 'organism': 'The video contains no person.'

            "The video contains multiple distinct shots, with significant changes in scenes or perspectives.": "The video contains multiple distinct shots, with significant changes in scenes or perspectives.",
            'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.': 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.',
            'Camera movement, the protagonist has small-scale body movements': 'Camera movement, the protagonist has small-scale body movements',
            'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.': 'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.',
            'The protagonist remain stationary in the video with no apparent movement.': 'The protagonist remain stationary in the video with no apparent movement.',
            'organism': 'No person in Video.',
        }
        prompt = torch.cat(
            [clip.tokenize(t.format(prompt_mapping[c])) for i, t in enumerate(templates) for c in labels])
    else:
        raise ValueError(f'`mode` must be `official` or `self`, but got {mode}')
    return prompt, num_prompt


@MODELS.register_module()
class ActionClip(BaseModel):

    def __init__(self,
                 clip_arch: str,
                 num_adapter_segs: int,
                 num_adapter_layers: int = 6,
                 to_float32: bool = False,
                 labels_or_label_file: Optional[Union[List[str], str]] = None,
                 templates_or_template_file: Optional[Union[List[str],
                                                            str]] = None,
                 data_preprocessor: Optional[Dict] = None,
                 loss: Dict = dict(type='CrossEntropyLoss', loss_weight=0.5)):
        super(ActionClip, self).__init__(data_preprocessor=data_preprocessor)
        self.clip = clip.load(clip_arch, device='cpu')[0]
        if to_float32:
            self.clip.float()

        self.adapter = TransformerAdapter(self.clip, num_adapter_segs,
                                          num_adapter_layers)

        self.loss = MODELS.build(loss)

        if labels_or_label_file is not None:
            self.prompt, self.num_prompt = text_prompt(
                labels_or_label_file, templates_or_template_file, mode='self')
            
        self.count = self.ccount = 0

    def encode_video(self, video):
        b, n, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        frames_features = self.encode_image(video)
        frames_features = frames_features.view(b, n, -1)
        video_features = self.adapter(frames_features)
        return video_features

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List] = None,
                mode: str = 'tensor'):

        if mode == 'tensor':
            return self.encode_video(inputs)

        elif mode == 'predict':
            assert hasattr(self, 'prompt'),\
                '`labels_or_label_file` is required to perform prediction. '

            video_features = self.encode_video(inputs)
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)

            bsz = len(data_samples)
            num_views = video_features.shape[0] // bsz

            text_features = self.encode_text(self.prompt.to(inputs.device))
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)

            # (bsz*num_views, num_prompt, num_classes) ->
            # (bsz, num_views*num_prompt, num_classes)
            similarity = (100.0 * video_features @ text_features.T). \
                view(bsz, num_views * self.num_prompt, -1)
            
            cls_scores = F.softmax(similarity, dim=2).mean(dim=1)

            for data_sample, score in zip(data_samples, cls_scores):
                # 输出时映射到[0,1]二分类
                # print(data_sample.gt_label[0], data_sample.gt_label)
                data_sample.gt_label = torch.tensor([int(data_sample.gt_label[0] == 1)], device=score.device)
                score = torch.tensor([sum(score[0:1])+sum(score[2:]), score[1]], device=score.device)

                data_sample.pred_scores = LabelData(item=score)
                # with open("/workspace/val.txt", "a") as f:
                #     row = data_sample.pred_scores.item.cpu().numpy()
                #     row_str = ",".join(map(str, row.tolist()))
                #     f.write(row_str + "\n")
            return data_samples

        elif mode == 'loss':
            video_features = self.encode_video(inputs)
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)

            text_id = np.random.randint(
                self.num_prompt, size=len(data_samples))
            real_labels = [x.gt_label.item() for x in data_samples]
            selected_prompt = self.prompt.view(
                self.num_prompt, -1,
                self.prompt.shape[-1])[text_id, real_labels].to(inputs.device)

            text_features = self.encode_text(selected_prompt)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)

            video_features = torch.cat(
                GatherLayer.apply(video_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            logit_scale = self.clip.logit_scale.exp()
            logits_per_video = logit_scale * video_features @ text_features.t()
            logits_per_text = logits_per_video.t()
            labels = torch.arange(logits_per_video.shape[0]).to(logit_scale.device)
            # labels = torch.tensor(real_labels).to(logit_scale.device)
            sim_loss_v2t = self.loss(logits_per_video, labels)
            sim_loss_t2v = self.loss(logits_per_text, labels)

            losses = dict()
            losses['sim_loss_v2t'] = sim_loss_v2t
            losses['sim_loss_t2v'] = sim_loss_t2v
            return losses

        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". '
                'Only supports `predict`, `loss` and `tensor` mode. ')