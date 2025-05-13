import av
import numpy as np
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification
import sys, os
import csv
from tqdm import tqdm
import json
import cv2
import requests
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_feature_extractor
from models.blip_pretrain_text import blip_pretrain as blip_pretrain_text
from models.blip_pretrain import blip_pretrain as blip_pretrain_img
import torchvision.models as models
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from glob import glob
from models.load import init_actionclip
from mmaction.utils import register_all_modules
from mmaction.apis import inference_recognizer
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose
import clip
import ruamel.yaml as yaml
from character.samurai.samurai_with_gdino_t2v import infer_groundino
from character.BLIP.infer_retrieval_pretrain_v2 import create_gallery, create_query, compute_gallery_embeds, evaluation
from character.BLIP.blip_models.blip_pretrain import blip_pretrain
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _read_video_pyav(
    frame_paths:List[str], 
    max_frames:int,
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from quality dimensions: the quality of the video in terms of clearness, resolution, brightness, and color

Output a float number from 0 to 100,
the higher the number is, the better the video performs in that sub-score, 
the lowest 0 means Bad, the highest 100 means Perfect/Real (the video is like a real video)

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

ROUND_DIGIT=3

MAX_NUM_FRAMES=24
MAX_LENGTH = 144
IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quality_model_name = "./weights/mantis-8b-idefics2-video-eval-24frame-quality_4096_regression/checkpoint-final"
smooth_model_name = "./weights/mantis-8b-idefics2-video-eval-24frame-benchmark-1epoch_4096_regression/checkpoint-final"

# load quality model
quality_processor = AutoProcessor.from_pretrained(quality_model_name,torch_dtype=torch.bfloat16)
quality_model = Idefics2ForSequenceClassification.from_pretrained(quality_model_name,torch_dtype=torch.bfloat16).eval()
quality_model.to(device)

# load smooth model
smooth_processor = AutoProcessor.from_pretrained(smooth_model_name,torch_dtype=torch.bfloat16)
smooth_model = Idefics2ForSequenceClassification.from_pretrained(smooth_model_name,torch_dtype=torch.bfloat16).eval()
smooth_model.to(device)

# load img-video consistency model
img_video_con_name = "./weights/Pretrain_humangt_meanvideo_trainset_wGT/checkpoint_06.pth"
img_video_con_model = blip_pretrain_img(pretrained=img_video_con_name, image_size=IMAGE_SIZE, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, queue_size=57600)
img_video_con_model.eval()
img_video_con_model = img_video_con_model.to(device)

# load text-video consistency model
text_video_con_name = "./weights/Pretrain_text_humangt_meanvideo_trainset_wGT_random_samescore/checkpoint_05.pth"
text_video_con_model = blip_pretrain_text(pretrained=text_video_con_name, image_size=IMAGE_SIZE, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, queue_size=57600)
text_video_con_model.eval()
text_video_con_model = text_video_con_model.to(device)

# load motion model
register_all_modules(True)
motion_model_name = "./weights/testvl-pre76-top187-rec69.pth" 
motion_model, motion_preprocess = init_actionclip(motion_model_name, device=device, mode='self')

# load character model
config = './character/BLIP/blip_configs/eval_pretrain.yaml'
yaml = yaml.YAML(typ='rt')
with open(config, 'r') as config_file:
    config = yaml.load(config_file)
character_model_name = './character/IP_checkpoints/blip_checkpoints/animate_recoginzer.pth'
character_model = blip_pretrain(pretrained=character_model_name, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
character_model.eval() 
character_model = character_model.to(device) 
gallery_folder = './character/videos'
frame_nums = 8 
gallery_loader = create_gallery(config, frame_nums , gallery_folder)
gallery_embeds = compute_gallery_embeds(character_model, gallery_loader, device)

success_count = 0
total_scores = []

def prepare_text_embedding(device):
    # for 2 types infer:
    PROMPTs = ['The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.',
               'The protagonist remain stationary in the video with no apparent movement.']

    # for 6 types infer:
    # PROMPTs = [ 'The protagonist has a large range of movement, such as running, jumping, dancing, or waving arms.',
    #             'The video contains multiple distinct shots, with significant changes in scenes or perspectives.',
    #             'Camera movement, the protagonist has small-scale body movements',
    #             'The protagonist performs some facial movements, such as changes in expressions, eye movements, or mouth movements.',
    #             'The protagonist remain stationary in the video with no apparent movement.',
    #             'No person in Video.']
    templates = ['{}']
    num_prompt = 1 #len(templates)
    text = torch.cat([clip.tokenize(templates[0].format(c)) for c in PROMPTs]).to(device)
    return text

def load_key_image(image_size,img_path, device):
    raw_image = img_path
    if isinstance(img_path, str): 
        raw_image = Image.open(img_path).convert('RGB')

    w,h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

with torch.no_grad():
    motion_text = prepare_text_embedding(device)
text_features = motion_model.encode_text(motion_text)

video_prompt="In this video, a group of samurai are depicted in a forest at night, engaged in a confrontation. The samurai are dressed in traditional armor and carry swords, indicating a historical or fantasy setting. The scene is filled with tension as the characters face each other, their expressions and body language suggesting a high-stakes encounter. The forest backdrop, illuminated by the light of the samurai's torches, adds a dramatic and mysterious atmosphere to the scene. The video captures the essence of a classic samurai story, with elements of honor, betrayal, and the clash of ideologies."

img_path = "asset/image/cartoon_films_da_dou_1.png"    
video_path = "asset/video/cartoon_films_da_dou_1.mp4"
output_path = "result"    
os.makedirs(output_path,exist_ok=True) 
key_image = load_key_image(image_size=IMAGE_SIZE, img_path=img_path, device=device)

#try:
# sample uniformly 8 frames from the video
container = av.open(video_path)
total_frames = container.streams.video[0].frames
if total_frames > MAX_NUM_FRAMES:
    indices = np.arange(0, min(total_frames,MAX_LENGTH), min(total_frames,MAX_LENGTH) / MAX_NUM_FRAMES).astype(int)
else:
    indices = np.arange(total_frames)
frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
num_image_token = eval_prompt.count("<image>")
if num_image_token < len(frames):
    eval_prompt += "<image> " * (len(frames) - num_image_token)

flatten_images = []

for x in [frames]:
    if isinstance(x, list):
        flatten_images.extend(x)
    else:
        flatten_images.append(x)

img_all = []
for x in flatten_images:
    img = load_key_image(image_size=IMAGE_SIZE, img_path=x, device=device)
    img_all.append(img.unsqueeze(0))
img_all = torch.cat(img_all, 1)

indices = torch.linspace(0, len(frames)-1, steps=8).long()
with torch.no_grad():
    video_features = motion_model.encode_video(img_all[:,indices,:,:,:])
video_features /= video_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
probs = (100.0 * video_features @ text_features.T).softmax(dim=-1)
motion_score = probs.detach().cpu()[0,0].item()

ivc_sim_out = img_video_con_model.inference(img_all, video_prompt, key_image)
tvc_sim_out = text_video_con_model.inference(img_all, video_prompt, key_image)
ivc_score = []
ivc_score.append(ivc_sim_out.cpu().detach().numpy())
ivc_sim = np.array(ivc_score)
ivc_sim = float(np.mean(ivc_sim))

tvc_score = []
tvc_score.append(tvc_sim_out.cpu().detach().numpy())
tvc_sim = np.array(tvc_score)
tvc_sim = float(np.mean(tvc_sim))


flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
quality_inputs = quality_processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
quality_inputs = {k: v.to(quality_model.device) for k, v in quality_inputs.items()}

smooth_inputs = smooth_processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
smooth_inputs = {k: v.to(smooth_model.device) for k, v in smooth_inputs.items()}

with torch.no_grad():
    quality_outputs = quality_model(**quality_inputs)
    smooth_outputs = smooth_model(**smooth_inputs)

quality_logits = quality_outputs.logits
num_aspects = quality_logits.shape[-1]

quality_aspect_scores = []
for i in range(num_aspects):
    quality_aspect_scores.append(round(quality_logits[0, i].item(),ROUND_DIGIT))
quality_score = quality_aspect_scores[0]/100

smooth_logits = smooth_outputs.logits
num_aspects = smooth_logits.shape[-1]

smooth_aspect_scores = []
for i in range(num_aspects):
    smooth_aspect_scores.append(round(smooth_logits[0, i].item(),ROUND_DIGIT))
smooth_score = smooth_aspect_scores[0]/100

character_save_dir = "character_save"
infer_groundino(video_path, character_save_dir, device)
config['query'] = os.path.join(character_save_dir,os.path.basename(video_path).split('.')[0])
query_loader = create_query(config, frame_nums)
ip_score = evaluation(character_model, gallery_loader, gallery_embeds, query_loader, frame_nums, device)

results_dict={
    "visual_smooth": smooth_score,
    "visual_motion": motion_score,
    "visual_quality": quality_score,
    "text-video_consist": tvc_sim,
    "img-video_consist": ivc_sim,
    "character_consist": ip_score,
}
print(results_dict)
output_name = os.path.join(output_path,'eval_results.json')
with open(output_name, 'w') as json_file:
    json.dump(results_dict, json_file)
print(f'Evaluation results saved to {output_name}')
