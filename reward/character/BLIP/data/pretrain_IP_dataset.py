import json
import os
import random
import torch
from torch.utils.data import Dataset
from decord import VideoReader
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from .utils import pre_caption
import os,glob
import cv2
from scipy.stats import entropy

def resambling(name: str, count = 1):
    templates = 'The person captured here is recognized as {name}.'.replace('{name}', name)
    return templates

def load_rgb_frames(object_id, num, masks_inputs, image_folder_path):  
    image_paths = sorted([os.path.join(image_folder_path, i) for i in os.listdir(image_folder_path)])
    
    masks, frames = [], []
    assert len(image_paths) > 0
    assert len(masks_inputs) > 0
    for image_path, mask in zip(image_paths, masks_inputs):
        mask_i = (mask == object_id).astype(np.uint8)
        if mask_i.max() > 0:
            masks.append(mask_i)
            frames.append(image_path)
            
    step = len(frames) // num
    if step == 0:
        return None, None

    frames = frames[::step][:num]
    masks = masks[::step][:num]

    images = []
    masks_ = []
    for i, (image_path, mask_i) in enumerate(zip(frames, masks)):
        image = cv2.imread(image_path)#[:, :, [2, 1, 0]]
        y_indices, x_indices = np.where(mask_i > 0)
        try:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
        except:
            print(mask_i, image_path)

        cropped_frame = image[y_min:y_max+1, x_min:x_max+1, :]
        cropped_mask = mask_i[y_min:y_max+1, x_min:x_max+1]
        image, mask_i = cropped_frame, cropped_mask
        # image = image * mask_i[..., np.newaxis] # fnum, [h,w,3]
        images.append(image)
        masks_.append(mask_i)
    
    # crop to square   
    # cropped_height = max([i.shape[0] for i in images])
    # cropped_width = max([i.shape[1] for i in images])
    # side_length = max(cropped_height, cropped_width)
    square_vid_tube = []
    square_masks = []
    for i, (img, mask) in enumerate(zip(images, masks_)):
        cropped_height, cropped_width = img.shape[0], img.shape[1]
        side_length = max(cropped_height, cropped_width)
        pad_top = (side_length - cropped_height) // 2
        pad_bottom = side_length - cropped_height - pad_top
        pad_left = (side_length - cropped_width) // 2
        pad_right = side_length - cropped_width - pad_left
        vid_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask_padded = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0])
        square_vid_tube.append(vid_padded[:,:,::-1])
        square_masks.append(mask_padded)
        
    assert len(images) > 0
    return square_vid_tube, square_masks#

class RetrievalVideoDataset(Dataset):

    def __init__(self, ann_file, transform, max_img_size=224, mode='query', frame_nums=4, gallery_prefix=''):
        super().__init__()
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''                
        if os.path.isdir(ann_file) or mode == 'infer':
            print(f'Infer videos from {ann_file}')
            annotation_dict = {}
            self.annotation = []
            videos_num = fail = 0
            
            image_folder_path = f'{ann_file}/frames'
            mask_path = f'{ann_file}/masks/mask.npy'
            if not os.path.exists(mask_path):
                fail += 1
            try:
                object_ids = list(set([int(i.split('_')[0][-1]) for i in os.listdir(f'{ann_file}/mask_crop')]))
            except:
                masks = np.load(mask_path, 'r')
                object_ids = int(masks.max())
                if object_ids <= 0 or masks is None:
                    fail += 1
                object_ids = range(1, object_ids+1)
            mask = []
            flag = 0
            for object_id in object_ids:
                mask_crop_exists = True
                mask_paths = []
                for i in range(frame_nums):
                    save_mask_path = f'{ann_file}/mask_crop/object{object_id}_img{i}.jpg'
                    if not os.path.exists(save_mask_path): 
                        mask_crop_exists = False 
                        break
                    else:
                        mask_paths.append(save_mask_path)
                if mask_crop_exists: 
                    self.annotation.append({'video': mask_paths})
                    flag = 1
                else:
                    _imgs, _masks = load_rgb_frames(object_id, frame_nums, masks, image_folder_path)
                    if _imgs is not None and _masks is not None:
                        mask_paths = []
                        for i in range(len(_imgs)):
                            masked_image = _imgs[i] * _masks[i][..., np.newaxis]
                            save_mask_path = f'{ann_file}/mask_crop/object{object_id}_img{i}.jpg'
                            if not os.path.exists(os.path.dirname(save_mask_path)): os.mkdir(os.path.dirname(save_mask_path))
                            cv2.imwrite(save_mask_path, masked_image[:,:,::-1])
                            flag = 1
                            mask_paths.append(save_mask_path)
                        self.annotation.append({'video': mask_paths})
                    else:
                        fail += 1
            if flag: videos_num += 1
            #print(videos_num, 'videos compared in', len(os.listdir(ann_file)), 'videos.', fail, 'lack data.')
            print('loading', len(self.annotation), 'videos in query.')
            self.mode = 'infer'
            self.max_img_size = max_img_size
            self.transform = transform
            
        elif ann_file.endswith('.json'):
            annotation = json.load(open(ann_file, 'r', encoding='utf-8'))
            if mode == 'gallery':
                self.annotation = annotation  
                self.anno_prefix = gallery_prefix
                print('loading', len(self.annotation), 'images in gallery.') 
            elif mode == 'query':
                annotation_dict = {}
                caption = {}
                video_name = {}
                for data in annotation:
                    name = data['image'].split('/')[-3] + '_' + data['caption']
                    if name not in video_name:
                        video_name[name] = 1
                        annotation_dict[name] = [data['image']]
                        caption[name] = data['caption']
                    else:
                        annotation_dict[name].append(data['image'])
                
                self.annotation = []
                for k, v in annotation_dict.items():
                    if len(annotation_dict[k]) == frame_nums:
                        # print(len(annotation_dict[k]), k)
                        self.annotation.append({'video': v, 'caption': caption[k]})
                    elif len(annotation_dict[k]) > frame_nums:
                        annotation_k = random.sample(annotation_dict[k], k=frame_nums)
                        self.annotation.append({'video': annotation_k, 'caption': caption[k]})
                    else:
                        print(k, len(annotation_dict[k]))
                print('loading', len(self.annotation), 'videos in query.') 
            else:
                assert mode in ['query', 'gallery']
            self.mode = mode
            self.max_img_size = max_img_size
            self.transform = transform
            self.text = list(set([pre_caption(ann['caption'],40) for ann in self.annotation]))
            print(len(self.text), 'IDs in datasets.')
            self.txt2video = [i for i in range(len(self.annotation))]
            self.video2txt = [self.text.index(pre_caption(i['caption'])) for i in self.annotation]#self.txt2video     
            self.names = list([ann['caption'] for ann in self.annotation])
        else:
            assert f'check {ann_file}'
            
            
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]  

        if self.mode == 'query':
            video_path = ann['video']
            vid_frm_array = self._load_video_from_path(video_path, height=self.max_img_size, width=self.max_img_size)
            video = self.transform(vid_frm_array)
            return video, ann['caption'], video_path
        
        elif self.mode == 'gallery':
            img_path = os.path.join(self.anno_prefix, ann['image'])
            image = Image.open(img_path).convert('RGB')   
            image = self.transform(image)
            caption = pre_caption(ann['caption'],30)
            
            return image, caption
        
        elif self.mode == 'infer':
            video_path = ann['video']
            vid_frm_array = self._load_video_from_path(video_path, height=self.max_img_size, width=self.max_img_size)
            video = self.transform(vid_frm_array)
            
            return video, video_path

    def _load_video_from_path(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        raw_sample_frms = []
        for path in video_path:
            img = np.array(Image.open(path, 'r').convert('RGB').resize((224, 224), Image.BILINEAR))
            # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            # print(img.shape)
            raw_sample_frms.append(img)
        raw_sample_frms = np.stack(raw_sample_frms)
        raw_sample_frms = torch.from_numpy(raw_sample_frms).contiguous().permute(0, 3, 1, 2).to(dtype=torch.float).div(255)
        
        
        return raw_sample_frms
    
if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    transform_test = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])  
    d = RetrievalVideoDataset('/workspace/BLIP/data/query_1031.json', transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), mode='query')
    v, c, p  = d.__getitem__(13)
    print(v.shape, c, p)
