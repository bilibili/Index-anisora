# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import cv2
import json
import datetime
import utils.imgproc as imgproc
from moviepy import *


def is_video_path(string):
    video_extensions = ['.mp4', '.avi', '.mov']
    return any(string.lower().endswith(ext) for ext in video_extensions)

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def find_video_files(folder_path):
    mp4_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.MP4') :
                mp4_files.append(os.path.join(root, file))
    print(f"Load {len(mp4_files)} videos from {folder_path}")
    return mp4_files

def read_csv_data(csv_file):
    data = {}
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            avid_cid = f"{row['avid']}_{row['cid']}"
            data[avid_cid] = row['season_type']
    return data

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def read_json(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    print(f"Load {len(json_data)} json data from {json_path} !")
    return json_data

def write_json(json_data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file)
    print(f"Save {len(json_data)} json data to {json_path} !")


def format_seconds(seconds):
    td = datetime.timedelta(seconds=seconds)
    formatted_time = str(td).split(".")[0]
    return formatted_time

def get_video_duration(video_path):
    total_duration = 0
    try:
        clip = VideoFileClip(video_path)
        total_duration = clip.duration
        clip.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    return total_duration


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
    return frame_info

def crop_video(in_vid, out_vid, crop_box_coor):
    video_capture = cv2.VideoCapture(in_vid)

    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(out_vid, fourcc, fps, (width, crop_box_coor[1]))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        cropped_frame = frame[:crop_box_coor[1], :]
        
        output_video.write(cropped_frame)

    video_capture.release()
    output_video.release()

    print(f"cropped video saved in {out_vid}")
