import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
import json,traceback

VIDEO_size = {"480p": [480, 720], "640p": [640, 960], "720p": [720, 1088], "1080p": [1088, 1632]}

def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


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


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if tensor.shape[0] < num_frames:
        last_frame = tensor[-int(num_frames - tensor.shape[1]) :]
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item


class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)


class SFTDataset(Dataset):
    def __load_video_info__(self, jsonfile, video_base_dir, fps, bucket_config, skip_frms_num):
        with open(jsonfile, 'r') as file:
            json_data = json.load(file)
        if 'val' in jsonfile:
            self.val = True
        else:
            self.val = False
        # print('self.val', self.val)
        
        self.vs = []
        self.min_num_frames = 999999999
        for size, F_dict in bucket_config.items():
            video_size = VIDEO_size[size]
            for num_frames, p in F_dict.items():
                if p>0:
                    if num_frames<self.min_num_frames:
                        self.min_num_frames = num_frames
                    self.vs.append([video_size, num_frames])
                    
                    self.p[str([video_size, num_frames])] = 0
                    self.videos_list[str([video_size, num_frames])] = []
                    self.captions_list[str([video_size, num_frames])] = []
                    self.num_frames_list[str([video_size, num_frames])] = []
                    self.fps_list[str([video_size, num_frames])] = []
                    self.video_size_list[str([video_size, num_frames])] = []

                    self.w_score_list[str([video_size, num_frames])] = []
                    self.l_score_list[str([video_size, num_frames])] = []

                    # self.w_rank_list[str([video_size, num_frames])] = []
                    # self.l_rank_list[str([video_size, num_frames])] = []

        self.train_step = 0
        
        def find_nearest_smaller_nframe(n, fps_oir):
            n = int((n / fps_oir)*fps)+1
            max_item = None
            for item in self.vs:
                if item[1] <= n and (max_item is None or item[1] > max_item[1] or (item[1] == max_item[1] and item[0][0] > max_item[0][0])):
                    max_item = item
            return max_item
            
        for videoinfo in json_data:
            try:
                videoinfo['total_frames']=float(videoinfo['total_frames'])
                videoinfo['fps']=float(videoinfo['fps'])

                videoinfo['w_score']=float(videoinfo['w_score'])
                videoinfo['l_score']=float(videoinfo['l_score'])

                if (videoinfo['total_frames'] / videoinfo['fps']) < ( (self.min_num_frames)/fps ) :
                    continue
                ori_vlen = videoinfo['total_frames']
                video_path = videoinfo['vae_path']
                cho_item = [videoinfo['vae_HW'], videoinfo['vae_frame']]
                if cho_item[1]>0 and str(cho_item) in self.p.keys():
                    video_size , num_frames = cho_item
                    self.p[str(cho_item)] = self.p[str(cho_item)] + 1
                    self.videos_list[str(cho_item)].append(video_path)
                    self.captions_list[str(cho_item)].append(videoinfo['caption'])
                    self.num_frames_list[str(cho_item)].append(num_frames)
                    self.fps_list[str(cho_item)].append(fps)
                    self.video_size_list[str(cho_item)].append(video_size)

                    self.w_score_list[str(cho_item)].append(videoinfo['w_score'])
                    self.l_score_list[str(cho_item)].append(videoinfo['l_score'])
                else:pass
                    # print(video_path, 'is too short')
            except:
                traceback.print_exc()
            
        total_video_num = 0
        del_key = []
        for item in enumerate(self.p):
            key_ = item[1]
            if self.p[key_]<1:
                del_key.append(key_)
        for i in range(len(del_key)):
            key_ = del_key[i]
            del self.p[key_]
            del self.videos_list[key_]
            del self.captions_list[key_]
            del self.num_frames_list[key_]
            del self.fps_list[key_]
            del self.video_size_list[key_]
            self.vs.remove(eval(key_))
        
        for item in enumerate(self.p):
            key_ = item[1]
            total_video_num = total_video_num + self.p[key_]
        for item in enumerate(self.p):
            key_ = item[1]
            self.p[key_] = float(self.p[key_])/total_video_num
        self.sample_p = [x*100 for x in self.p.values()]
        self.sample_p_new = []
        for i in range(len(self.sample_p)):
            self.sample_p_new.append(sum(self.sample_p[:i+1]))
        
        self.val_len = 0
        for item in enumerate(self.videos_list):
            key_ = item[1]
            if len(self.fps_list[key_])>self.val_len:
                self.val_len = len(self.fps_list[key_])
            print('video num:', len(self.videos_list[key_]), 'cap num:', len(self.captions_list[key_]), 'frame num:', len(self.num_frames_list[key_]), 'fps num:', len(self.fps_list[key_]), 'shape', key_, 'p', self.p[key_])

    def __get_video__(self, video_path, video_size, num_frames, fps, skip_frms_num):
        video_path_w,video_path_l= video_path.split(',')
        latent_w=torch.load(video_path_w,map_location="cpu")[0] 
        latent_l=torch.load(video_path_l,map_location="cpu")[0]
        latent = torch.cat([latent_w, latent_l], dim=0)
        return latent 

    def __init__(self, data_dir, video_base_dir,  bucket_config, fps, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()
        
        self.p = {}
        self.videos_list = {}
        self.captions_list = {}
        self.num_frames_list = {}
        self.fps_list = {}
        self.video_size_list = {}
        self.skip_frms_num = skip_frms_num
        self.bucket_config = bucket_config
        
        self.w_score_list = {}  
        self.l_score_list = {} 

        self.w_rank_list = {}  
        self.l_rank_list = {}

        decord.bridge.set_bridge("torch")
        self.__load_video_info__(data_dir, video_base_dir, fps, bucket_config, skip_frms_num)
        self.cache=None


    def find_interval(self, a):
        for i in range(len(self.sample_p_new)):
            if a <= self.sample_p_new[i]:
                return i
        return f"{a} 不属于任何区间"
    
    def __getitem__(self, index):
        try:
            if self.val:
                index__ = np.random.choice(len(self.vs))
                key_ = str(self.vs[index__])
                index_ = index
                print('index_', index_)
            else:
                np.random.seed(self.__getstep__())
                index__ = np.random.choice(len(self.vs))
                key_ = str(self.vs[index__])
                np.random.seed()
                index_ = np.random.choice(range(len(self.videos_list[key_])), 1)[0]

            item = {
                "mp4": self.__get_video__(self.videos_list[key_][index_], self.video_size_list[key_][index_], self.num_frames_list[key_][index_], self.fps_list[key_][index_], self.skip_frms_num),
                "txt": self.captions_list[key_][index_],
                "num_frames": self.num_frames_list[key_][index_],
                "fps": self.fps_list[key_][index_],
                "w_score" : self.w_score_list[key_][index_],
                "l_score" : self.l_score_list[key_][index_],
                # "w_rank" : self.w_rank_list[key_][index_],
                # "l_rank" : self.l_rank_list[key_][index_]
            }
            self.cache=item
        except:
            traceback.print_exc()
            if (self.cache):
                item = self.cache
            else:
                return self.__getitem__(random.randint(0, len(self.fps_list) - 1))
        self.updatestep()
        return item

    def updatestep(self):
        self.train_step = self.train_step + 1
    
    def __getstep__(self):
        return self.train_step
        
    def __len__(self):
        if self.val:
            return self.val_len
        return len(self.fps_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

if __name__ == "__main__":
    bucket_config={
        # ---480 720
        "480p":  {9: 0, 17: 0, 25: 0, 33: 0, 41: 1, 49: 1, 57: 0, 65: 0, 73: 1}, # 73only-ok
        # ---640 960
        # "640p":  {9: 0, 17: 0, 25: 0, 33: 0, 41: 0, 49: 1, 57: 0, 65: 0}, 
        # # ---720 1088
        # "720p":  {9: 0, 17: 0, 25: 0, 33: 0, 41: 1, 49: 0, 57: 0, 65: 0}, 
        # # ---1080 1632
        # "1080p": {9: 0, 17: 0, 25: 0, 33: 0, 41: 0, 49: 0, 57: 0, 65: 0},
    }