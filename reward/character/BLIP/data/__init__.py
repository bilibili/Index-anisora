import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .pretrain_IP_dataset import RetrievalVideoDataset
from .sampler import RandomIdentitySampler

def create_dataset(mode, config, frame_nums=4, min_scale=0.5, gallery_prefix = ''):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    transform_eval = transforms.Compose([
        normalize,
    ])
    if mode == 'gallery':
        dataset = RetrievalVideoDataset(config['gallery'], transform_test, mode=mode, gallery_prefix = gallery_prefix)
    if mode == 'query':
        dataset = RetrievalVideoDataset(config['query'], transform_eval, mode=mode, frame_nums=frame_nums)
    
    return dataset
    
    
def create_sampler(datasets, mode, batch_size, shuffles=[False], num_tasks=1, global_rank=0):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        if mode == 'train':
            sampler = RandomIdentitySampler(data_source=dataset.data_source, batch_size=batch_size, num_instances=1)
        elif mode == 'eval' or mode == 'test':
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

