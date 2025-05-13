import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from moviepy import *
from core.utils import flow_viz
import clip
from net.net import MLP

def is_video_path(string):
    video_extensions = ['.mp4', '.avi', '.mov']
    return any(string.lower().endswith(ext) for ext in video_extensions)


def find_video_files(folder_path):
    mp4_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.MP4') :
                mp4_files.append(os.path.join(root, file))
    print(f"Load {len(mp4_files)} videos from {folder_path}")
    return mp4_files


def read_json(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    print(f"Load {len(json_data)} json data from {json_path} !")
    return json_data


def write_json(json_data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file)
    print(f"Save {len(json_data)} json data to {json_path} !")


def load_image(imfile, device='cuda'):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def load_frame(frame, device='cuda'):
    frame_np = frame.astype(np.uint8)
    frame_tensor = torch.from_numpy(frame_np)
    frame_tensor = frame_tensor.permute(2, 0, 1).float()
    return frame_tensor[None].to(device)


def resize_frame(frame, target_height=480):
    """
    Resize a frame while maintaining aspect ratio to make its height close to the target height.
    """
    height, width, _ = frame.shape
    scale_percent = target_height / height

    new_width = int(width * scale_percent)
    resized_frame = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_frame


def read_video_frames(video_path, frames_per_second=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    frame_info = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps

    if frames_per_second == 0:
        frames_per_second = fps

    interval = int(round(fps / frames_per_second))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0 or frame_idx == total_frames - 1:
            frame_info.append((frame, frame_idx))
        frame_idx += 1

    cap.release()

    # print(f"\nRead {len(frame_info)}/{duration_seconds}s frames from {video_path}")
    return frame_info


def viz(img, flo, save_file):
    img = img[0].permute(1,2,0).cpu().detach().numpy()
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    img_save = (img_flo[:, :, [2, 1, 0]] / 255.0 * 255).astype('uint8')
    cv2.imwrite(save_file, img_save)


def get_video_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

    return duration


def model_init(device='cuda', project_path=None):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    model_weights = "./weights/sac+logos+ava1-l14-linearMSE.pth"
    if project_path is not None:
        model_weights = os.path.join(project_path, model_weights)

    print(f"Load model from {model_weights}")
    s = torch.load(model_weights)   # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)

    model.to('cuda')
    model.eval()
    model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64

    return model, model2,  preprocess


def net_inference(pil_image, model, clip, process, device='cuda'):
    image = process(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())
    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    return prediction.cpu().detach().numpy()[0][0]

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
