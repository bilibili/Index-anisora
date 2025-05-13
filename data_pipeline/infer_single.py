import argparse
import sys,os
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2
import decord
import numpy as np
from PIL import Image
import statistics
from core.raft import RAFT
from core.utils.utils import InputPadder
from infer_utils import load_frame, viz, resize_frame
from infer_utils import is_video_path, find_video_files
from infer_utils import read_video_frames
from infer_utils import read_json, write_json
from infer_utils import model_init, net_inference

import utils.base_utils as base_utils
import utils.process_utils as process_utils
import utils.infer_utils as infer_utils

def init_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    current_path = os.path.abspath(__file__)
    project_path = os.path.dirname(current_path)
    model_path = os.path.join(project_path, './weights/raft-things.pth')
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to(args.device)
    model.eval()
    return model

def read_video(video_path):
    # input: video_path
    # output: frames_data, width, height
    # read video and extract more 
    video_reader = decord.VideoReader(video_path)
    #width, height = video_reader[0].shape[1], video_reader[0].shape[0]
    total_frames = len(video_reader)
    desired_frames = 16
    step = max(1, total_frames // desired_frames)
    sampled_indices = [i for i in range(0, total_frames, step)]
    #sampled_frames_data = [np.array(frames[i].asnumpy()) for i in sampled_indices]
    sampled_frames_data = [np.array(video_reader[i].asnumpy()) for i in sampled_indices]

    return sampled_frames_data

def column_means(matrix):
    return [statistics.mean(column) for column in zip(*matrix)]

def optical_flow_detect(model, in_vid, frames_per_secondond=1, save=False, device='cuda'):
    """
        * model: optical flow model
        * in_vid: input video path(.mp4)
        * frames_per_second: Number of frames sampled per second in the video, e.g. 0 means the default frame rate of the original video
        * save: Whether to retain the optical flow results, by default saved in the "result" folder 
        * device: cuda or cpu
    """
    video_name = os.path.splitext(os.path.basename(in_vid))[0]
    if save:
        current_path = os.path.abspath(__file__)
        project_path = os.path.dirname(current_path)
        result_folder = os.path.join(project_path, "./result", video_name)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

    video_optical_flow_info = []
    frame_optical_flow_list = []
    frame_info = read_video_frames(in_vid, frames_per_secondond)
    if frame_info:
        for idx in range(1, len(frame_info)):
            pre_frame, frame_idx1 = frame_info[idx - 1]
            cur_frame, frame_idx2 = frame_info[idx]

            pre_frame_resize = resize_frame(pre_frame)
            cur_frame_resize = resize_frame(cur_frame)

            pre_frame_tensor = load_frame(pre_frame_resize, device)
            cur_frame_tensor = load_frame(cur_frame_resize, device)

            padder = InputPadder(pre_frame_tensor.shape)
            pre_frame_tensor, cur_frame_tensor = padder.pad(pre_frame_tensor, cur_frame_tensor)

            flow_low, flow_up = model(pre_frame_tensor, cur_frame_tensor, iters=20, test_mode=True)
            flow_max_move = np.max(np.abs(flow_up.cpu().detach().numpy()))
            video_optical_flow_info.append({
                "frame_idx": frame_idx1,
                "frame_optical_flow": float(flow_max_move)
            })
            frame_optical_flow_list.append(float(flow_max_move))

            if save:
                save_filename = "%06d" % (frame_idx1)
                save_file = f'{result_folder}/{save_filename}.jpg'
                viz(pre_frame_tensor, flow_up, save_file)

    video_optical_flow_avg = sum(frame_optical_flow_list)/len(frame_optical_flow_list) if len(frame_optical_flow_list) > 0 else 0

    return video_optical_flow_info, video_optical_flow_avg

def ocr_detect(args, net, refine_net, in_vid, result_folder):
    video_name = os.path.splitext(os.path.basename(in_vid))[0]
    if args.save:
        current_path = os.path.abspath(__file__)
        project_path = os.path.dirname(current_path)
        result_folder = os.path.join(project_path, "./result", video_name)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

    video_ocr_info = []
    frame_ocr_list = []
    frame_info = base_utils.read_video_frames(in_vid, args.frames_per_second)

    if frame_info:
        for idx, (frame, frame_idx) in enumerate(frame_info):
            bboxes, polys, score_text = infer_utils.net_inference(net, frame, args.text_threshold, args.link_threshold, args.low_text, args.device, args.poly, args.canvas_size, args.mag_ratio, refine_net, args.show_time)

            frame_box_sum_percentage = process_utils.compute_sum_bbox_pct(frame[:,:,::-1], polys)
            frame_box_coordinate = [poly.astype(int).tolist() for poly in polys]
            # max_box_percentage = file_utils.compute_max_bbox_percentage(frame[:,:,::-1], polys)

            video_ocr_info.append({
                "frame_idx": frame_idx,
                "frame_text_sum_percentage": frame_box_sum_percentage,
                "frame_text_coordinate": frame_box_coordinate
            })
            frame_ocr_list.append(frame_box_sum_percentage)

            if args.save:
                mask_file = result_folder + "/res_" + "%06d" % frame_idx + '_mask.jpg'
                cv2.imwrite(mask_file, score_text)

                image_path = "%s/%06d.jpg" % (result_folder, frame_idx)
                process_utils.saveResult(image_path, frame[:,:,::-1], polys, None, dirname=result_folder)

    video_ocr_avg = sum(frame_ocr_list)/len(frame_ocr_list) if len(frame_ocr_list) > 0 else 0

    return video_ocr_info, video_ocr_avg

def cal_aes_score(model, clip_model, process, device, frame_data):
    shot_aesthetic_list = []
    for frame in frame_data:
        frame_pil = Image.fromarray(frame)
        prediction = net_inference(frame_pil, model, clip_model, process, device)
        aesthetic_pre = round(float(prediction), 5)
        shot_aesthetic_list.append(aesthetic_pre)
    
    aes_score = sum(shot_aesthetic_list)/len(shot_aesthetic_list)
    return aes_score

def cal_bb_score(model, frame_data):
    bb_score_list = []
    bb_size_list = []

    for index, frame in enumerate(frame_data):
        det_res = model(frame, verbose=False)
        max_area_ratio = 0
        bb_size = []
        for res in det_res:
            bboxes = res.boxes
            for bbox in bboxes:
                xywhn = bbox.xywhn.cpu().squeeze(0).tolist()
                area_ratio = xywhn[2] * xywhn[3]
                if area_ratio > max_area_ratio:
                    max_area_ratio = area_ratio
                    bb_size = xywhn
        bb_size_list.append(bb_size)
        bb_score_list.append(max_area_ratio)
    bb_score = sum(bb_score_list) / len(bb_score_list)
    bb_size_list = column_means(bb_size_list)
    return bb_score, bb_size_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help="device")
    parser.add_argument('--in_vid', default='sample/918195104_1325613009-Scene-008.mp4', help="input video path or video folder")
    parser.add_argument('--flow_threshold', type=float, default=3.0, help='optical flow move threshold')
    parser.add_argument('--frames_per_second', type=float, default=0, help='extract frame from video')
    parser.add_argument('--delete', default=False, type=bool, help='if delete still input video')
    parser.add_argument('--save', default=False, type=bool, help='if save flow image')
    parser.add_argument('--result_folder', default='./result', type=str, help='result saved path')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--trained_model', default='./weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    #parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    #parser.add_argument('--result_folder', default='./result', type=str, help='result save path folder')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='./weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--min_duration_sec', default=1, type=float, help='min video duration second')

    # images
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')

    # video process
    parser.add_argument('--box_percentage_threshold', default=1, type=float, help='threshold to filter large percentage text video')
    parser.add_argument('--label_name', default='test', type=str, help='dataset label for save label percentage path')
    parser.add_argument('--multi_process', default=False, action='store_true', help='if use multiprocess to process videos')
    parser.add_argument('--multi_num', default=6, type=int, help='multiprocess num')
    args = parser.parse_args()
    #video_path = "sample/917990389_1319574086-Scene-006.mp4"
    video_path = args.in_vid
    #area_thre = 0.9

    ## black bound detection
    bb_model = YOLO('./weights/black_bound_0122_v5.pt')
    frms = read_video(video_path)
    score, size = cal_bb_score(bb_model, frms)
    print(video_path, score)

    ## opticalflow detection
    flow_model = init_model(args)
    video_optical_flow, video_optical_flow_avg = optical_flow_detect(flow_model, video_path, args.frames_per_second, args.save, args.device)
    print(video_path, video_optical_flow, video_optical_flow_avg)

    ## ocr detection
    current_path = os.path.abspath(__file__)
    project_path = os.path.dirname(current_path)
    net, refine_net = infer_utils.init_model(args, project_path)
    video_text_pct, _ = ocr_detect(args, net, refine_net, video_path, args.result_folder)
    print(video_path, video_text_pct)

    ## aes model
    aes_model, clip_model, process = model_init(args.device)
    aes_score = cal_aes_score(aes_model, clip_model, process, args.device, frms)
    print(video_path, aes_score)
