import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
#sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)
    total_length = 0
    if osp.isdir(args.video_path):
        frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
        total_length = len(frames)
        frames = frames[args.first_frame_idx:]
        loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        height, width = loaded_frames[0].shape[:2]
    else:
        cap = cv2.VideoCapture(args.video_path)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]
        total_length = len(loaded_frames)
        loaded_frames = loaded_frames[args.first_frame_idx:]

        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video.")

    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, 30, (width, height))
    mask_video_path = os.path.join(os.path.dirname(args.video_output_path), 'mask.npy')
    if not os.path.exists(mask_video_path) or args.refresh_mask:
        mask_video = torch.zeros(total_length, height, width)
        mask_object_id = 1
    else:
        mask_video = np.load(mask_video_path)
        mask_object_id = mask_video.max() + 1

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True, start_frame_idx=args.first_frame_idx)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=mask_object_id)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                rows, cols = non_zero_indices[:, 0], non_zero_indices[:, 1]
                mask_video[frame_idx + args.first_frame_idx][rows, cols] = mask_object_id
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()
        np.save(mask_video_path, np.array(mask_video, dtype=np.uint16), allow_pickle=True, fix_imports=True)

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="character/IP_checkpoints/sam2_checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=False, help="Save results to a video.")
    parser.add_argument("--first_frame_idx", default=0, type=int, help="Define the txt_path represent which frame the object start to appear.")
    parser.add_argument("--refresh_mask", default=1, type=int, help="Refresh all the `mask.npy` file espcially for the `object_ids`.")
    args = parser.parse_args()
    main(args)
