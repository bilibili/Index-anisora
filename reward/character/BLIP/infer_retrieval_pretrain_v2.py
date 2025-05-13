
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from .blip_models.blip_pretrain import blip_pretrain
from . import blip_utils as utils
from .data import create_dataset, create_sampler, create_loader
from .metrics import eval_func

def group_gallery_according_to_names(similarity_tensor, names_list):
    from collections import defaultdict
    grouped_indices = defaultdict(list)
    for idx, name in enumerate(names_list):
        name = name.lower()
        grouped_indices[name].append(idx)
    merged_similarity = []
    final_names = []
    for name, indices in grouped_indices.items():
        merged_similarity.append(similarity_tensor[:, :, indices].mean(dim=-1))
        final_names.append(name)
    final_similarity_tensor = torch.stack(merged_similarity, dim=-1)
    return final_similarity_tensor, final_names

@torch.no_grad()
def evaluation(model, gallery_loader, gallery_embeds, query_loader,frame_nums, device):    
    print('Computing vision features for query...')
    texts = []
    query_embeds = []
    videos = []
    for video, video_path in query_loader: 
        B,N,C,W,H = video.size() 
        video = video.view(-1,C,W,H)
        
        video = video.to(device, non_blocking=True) 
        video_feat = model.visual_encoder(video)    
        video_embed = model.vision_proj(video_feat[:,0,:])   
        video_embed = video_embed.view(B,N,-1)      
        for bsi in range(B):
            vp = []
            for i in range(frame_nums):        
                vp.append(video_path[i][bsi])
            videos += [vp]
       
        query_embeds.append(video_embed)
    query_embeds = torch.cat(query_embeds, dim=0)    
    
    print('Retrievaling...')
    sims_matrix = query_embeds @ gallery_embeds.t()
    sims_matrix = sims_matrix.detach().cpu()
    output = []
    videos_score = {}
    sims_matrix = torch.tensor(sims_matrix)
    grouped_sims_matrix, grouped_names = group_gallery_according_to_names(sims_matrix, gallery_loader.dataset.names)
    softmax_scores = torch.nn.functional.softmax(grouped_sims_matrix, dim=-1)
    softmax_scores = softmax_scores.mean(dim=1)
    max_scores, indses = torch.topk(softmax_scores, dim=1, k=2)
    for index in range(len(softmax_scores)):
        video_name = videos[index][0].split('/')[-3]
        obj_dix = videos[index][0].split('/')[-1][6]
        if video_name not in videos_score:
            videos_score[video_name] = [max_scores[index][0].item()]
        else:
            videos_score[video_name].append(max_scores[index][0].item())
    for k, v in videos_score.items():
        value = sum(v) / len(v)
        output.append(value)
    return output[0]

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets_gallery, dataset_query = create_dataset('retrieval', config, frame_nums=args.frame_nums, min_scale=0.2, gallery_prefix=args.gallery_folder)
    datasets_gallery, dataset_query = [datasets_gallery], [dataset_query]

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    gallery_samplers = create_sampler(datasets_gallery, 'eval', 1, [False], num_tasks, global_rank)  
    query_samplers = create_sampler(dataset_query, 'eval', 1, [False], num_tasks, global_rank)  

    gallery_loader = create_loader(datasets_gallery, gallery_samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    query_loader = create_loader(dataset_query, query_samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    
    #### Model #### 
    print("Creating model")
    blip_path= '../IP_checkpoints/blip_checkpoints/animate_recoginzer.pth'
    model = blip_pretrain(pretrained=blip_path, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
    model.eval() 
    model = model.to(device)   
    evaluation(model, gallery_loader, query_loader, args.save_path, device, args.score_mode)

def create_gallery(config, frame_nums, gallery_folder):
    print("Creating gallery dataset...")
    datasets_gallery = create_dataset('gallery', config, frame_nums=frame_nums, min_scale=0.2, gallery_prefix=gallery_folder)
    datasets_gallery = [datasets_gallery]  
    gallery_samplers = create_sampler(datasets_gallery, 'eval', 1, [False])  

    gallery_loader = create_loader(datasets_gallery, gallery_samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    return gallery_loader

def create_query(config, frame_nums):
    print("Creating query dataset...")
    dataset_query = create_dataset('query', config, frame_nums=frame_nums)
    dataset_query = [dataset_query]       
    query_samplers = create_sampler(dataset_query, 'eval', 1, [False])  

    query_loader = create_loader(dataset_query, query_samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    return query_loader

@torch.no_grad()
def compute_gallery_embeds(model, gallery_loader, device):
    print('Computing vision features for gallery...')
    gallery_embeds = []
    for image, caption in gallery_loader:
        B, C, W, H = image.size()
        image = image.to(device, non_blocking=True)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        gallery_embeds.append(image_embed)
    gallery_embeds = torch.cat(gallery_embeds,dim=0)
    return gallery_embeds


def text2image_retrieval(video_embeds, text_embeds, mode, device, text_sample, texts_gt):
    video_embeds = video_embeds.to(device)
    text_embeds = text_embeds.to(device)
    
    sims_matrix = text_embeds @ video_embeds.t()
    assert len(texts_gt) == sims_matrix.shape[1]
    for i, sim in enumerate(sims_matrix):
        sim = list(sim)
        idx = sim.index(max(sim))
        print(text_sample[i], texts_gt[idx])
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='blip_configs/eval_pretrain.yaml')
    parser.add_argument('--output_dir', default='character/IP_checkpoints/blip_checkpoints')  
    parser.add_argument('--checkpoint', default='animate_recoginzer.pth')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:29501', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_mode', default='video', type=str) # video
    parser.add_argument('--cos_sim_only', default=1, type=int)
    parser.add_argument('--tsne', default=0, type=int)#0
    parser.add_argument('--frame_nums', default=8, type=int)
    parser.add_argument('--score_mode', default=1, type=int)#0
    parser.add_argument('--save_path', default='ip_consistency_ours.json', type=str)
    parser.add_argument('--input_folder', default='samurai/videos', type=str)
    parser.add_argument('--gallery_folder', default='character/videos', type=str)
    
    args = parser.parse_args()
    yaml = yaml.YAML(typ='rt')
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config['query'] = args.input_folder
    
    main(args, config)      
    