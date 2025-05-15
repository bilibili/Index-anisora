import time
from PIL import Image
import io

import torch
import numpy as np

def test_raw_reading(webds):
    it = iter(webds)
    t = time.time()
    for i in range(2048):
        tmp = next(it)['jpg']
    print("Raw: Time to read 2048 images: ", time.time() - t)

def test_jpg_reading(webds):
    it = iter(webds)
    t = time.time()
    for i in range(2048):
        tmp = next(it)['jpg']
        tmp1 = Image.open(io.BytesIO(tmp)).convert('RGB')
        # print(tmp1.mode)
        # # save
        # tmp1.save('test.jpg')
        # break
    print("JPG: Time to read 2048 images: ", time.time() - t)

from sat.data_utils import SimpleDistributedWebDataset, MetaDistributedWebDataset
def test_simpleds_reading(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
            yield tmp1

    d = SimpleDistributedWebDataset(path, process_fn, 0)
    it = iter(d)
    t = time.time()
    for i in range(2048):
        tmp = next(it)
    print("SimpleDS: Time to read 2048 images: ", time.time() - t)

def test_metads_reading(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
            yield tmp1

    d = MetaDistributedWebDataset(path, process_fn, 0)
    it = iter(d)
    t = time.time()
    for i in range(2048):
        tmp = next(it)
    print("metaDS: Time to read 2048 images: ", time.time() - t)

def test_metads_reading_tensor(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
            tmp1 = torch.from_numpy(np.array(tmp1)).permute(2, 0, 1)
            yield tmp1

    d = MetaDistributedWebDataset(path, process_fn, 0)
    it = iter(d)
    t = time.time()
    for i in range(2048):
        tmp = next(it)
    print("metaDS: Time to read 2048 images: ", time.time() - t)

def test_metads_dataloader(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB')
            # to tensor
            tmp1 = torch.from_numpy(np.array(tmp1)).permute(2, 0, 1)
            yield tmp1

    d = MetaDistributedWebDataset(path, process_fn, 0)
    batch_size = 1
    loader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=8)
    it = iter(loader)
    t = time.time()
    for i in range(2048 // batch_size):
        tmp = next(it)
    print("metaDS: Time to read 2048 images: ", time.time() - t)

def test_metads_resize(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB').resize((256, 256))
            # to tensor
            tmp1 = torch.from_numpy(np.array(tmp1)).permute(2, 0, 1)
            # resize
            # tmp1 = torch.nn.functional.interpolate(tmp1.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            yield tmp1

    d = MetaDistributedWebDataset(path, process_fn, 0)
    batch_size = 1
    loader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=4)
    it = iter(loader)
    t = time.time()
    for i in range(2048 // batch_size):
        tmp = next(it)
    print("metaDS: Time to read 2048 images: ", time.time() - t)


def test_metads_batch(path):
    def process_fn(stream):
        for x in stream:
            tmp1 = Image.open(io.BytesIO(x['jpg'])).convert('RGB').resize((256, 256))
            # to tensor
            tmp1 = torch.from_numpy(np.array(tmp1)).permute(2, 0, 1)
            yield tmp1

    d = MetaDistributedWebDataset(path, process_fn, 0)
    batch_size = 16
    loader = torch.utils.data.DataLoader(d, batch_size=batch_size, num_workers=8)
    it = iter(loader)
    t = time.time()
    for i in range(2048 // batch_size):
        tmp = next(it)
    print("metaDS: Time to read 2048 images: ", time.time() - t)

from webdataset import WebDataset
# test_raw_reading(WebDataset("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar"))
# test_jpg_reading(WebDataset("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar"))
# test_raw_reading(WebDataset("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar"))
# test_simpleds_reading("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar")
# test_metads_reading("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar")
# test_metads_dataloader("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar")
# test_metads_resize("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar")
test_metads_batch("/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_merged_cleaned_wds/part-00000/000000.tar")
# test_metads_batch("/mnt/shared/img_datasets/laion_high_resolution_imgs/part-00000/000000.tar")
# 2048 images