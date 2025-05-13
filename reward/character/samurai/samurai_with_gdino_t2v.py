import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import argparse
import numpy as np
from torchvision.transforms.functional import resize
import torchvision.transforms as TT
from torchvision.transforms import InterpolationMode
import sys

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def is_new_object(original_bboxes, new_bboxes, threshold = 0.2):
    to_append_bboxes = []
    for new_bbox in new_bboxes:
        flag = True
        for origin_bbox in original_bboxes.values():
            for _origin_bbox in origin_bbox:
                iou = calculate_iou(new_bbox, _origin_bbox)
                if iou >= threshold:
                    flag = False
        if flag: to_append_bboxes.append(new_bbox)
    if len(to_append_bboxes) == 0: return None
    return to_append_bboxes

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:  
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def filt_bboxes(bboxes, count_person):
    sorted_bboxes = sorted(bboxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    top_bboxes = sorted_bboxes[:5]
    count_person = 5
    return top_bboxes, count_person

def extract_frames(video_path, video_dir):
    os.makedirs(video_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        frame_filename = os.path.join(video_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    
def infer_groundino(video_path,video_save_dir,device,refresh_mask=1):
    video_name = os.path.basename(video_path).split('.')[0]
    video_dir = f"{video_save_dir}/{video_name}/frames"
    if not os.path.exists(video_dir):
        extract_frames(video_path, video_dir)
        
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    
    crop_height, crop_width, _  = cv2.imread(os.path.join(video_dir, frame_names[0])).shape

    # Person detection: |--Grounding-dino-base--|     {i+step}.jpg -> [x, y, w, h]
    ckpt_folder = "./character/IP_checkpoints"
    model_id = os.path.join(ckpt_folder, "grouding-dino-base")
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # load [first frame] from images/videos
    image_path = video_path.replace('video','image').replace('.mp4','.png')
    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
    if frame is not None:
        height, width = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        cropped_frame = resize_for_rectangle_crop(image, (crop_height, crop_width), reshape_mode="center")
        cropped_frame = np.array(cropped_frame.permute(1,2,0))  
        image = Image.fromarray(cropped_frame)
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
        
    inputs = processor(images=image, text='character.', return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.5,
        text_threshold=0.5,
        target_sizes=[image.size[::-1]]
    )
    bboxes = results[0]["boxes"].cpu().numpy().tolist()
    count_person = len(bboxes)
    if count_person > 5:
        bboxes, count_person = filt_bboxes(bboxes, count_person)
    print(count_person, f'person, {video_path}')
    
    bboxes_dir = f"{video_save_dir}/{video_name}/bboxes"
    os.makedirs(bboxes_dir,exist_ok=True)    
    person_idx = frame_idx = 0
    for person_idx, _bbx in enumerate(bboxes):
        image = cropped_frame
        cv2.rectangle(image, (round(_bbx[0]), round(_bbx[1])), (round(_bbx[2]), round(_bbx[3])), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(bboxes_dir, f'{frame_idx:05d}_{person_idx:03d}.jpg'), image)
        with open(os.path.join(bboxes_dir, f'{frame_idx:05d}_{person_idx:03d}.txt'), 'w') as f:
            f.writelines(f'{round(_bbx[0])},{round(_bbx[1])},{round(_bbx[2]-_bbx[0])},{round(_bbx[3]-_bbx[1])}')
        person_idx += 1
    assert count_person == person_idx
    
    # SAM 2 and track: |--Samurai--| [x, y, w, h] + video_dir/00***.jpg -> masks
    bboxes_names = [p for p in os.listdir(bboxes_dir) if p.endswith('.txt')]
    bboxes_names.sort(key=lambda p: int((os.path.splitext(p)[0]).split('_')[1]))
    for person_idx, bbox_name in enumerate(bboxes_names):
        mask_video_path = f"{video_save_dir}/{video_name}/masks"
        os.makedirs(mask_video_path, exist_ok=True)
        bbox_path = os.path.join(bboxes_dir, bbox_name)
        start_frame_idx = int(bbox_name.split('_')[0])
        cmd = f'python character/samurai/scripts/demo.py --video_path {video_path} \
--txt_path {bbox_path} --video_output_path {mask_video_path}/person_{person_idx:03d}.mp4 --first_frame_idx {start_frame_idx} --refresh_mask {refresh_mask}'
        os.system(cmd)
        print(f'mask saved in {mask_video_path}')
        refresh_mask = 0
