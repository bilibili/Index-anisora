# -*- encoding: utf-8 -*-
'''
@File    :   configure_data.py
@Time    :   2021/01/11 23:28:38
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import copy

import numpy as np
import torch
from bisect import bisect_right
from functools import partial

from torch.utils import data
from .samplers import DistributedBatchSampler
from torch.utils.data import ChainDataset, IterableDataset

from sat import mpu
from sat.helpers import print_all, print_rank0

def make_data_loader(dataset, batch_size, args, split, collate_fn=None):

    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    distributed = world_size > 1

    # if IterableDataset, assume everything is properly configured. (pre-sharded) 
    if isinstance(dataset, IterableDataset):
        if split in ['val', 'test'] and args.strict_eval:
            raise ValueError('IterableDataset cannot be used for validation or testing if `args.strict_eval=True`, because we cannot infer the length of the final batch before reading out them.')
        args.val_last_shape = [1] * world_size # just fake it, not actually used
        args.val_drop_number = 0
        args.test_last_shape = [1] * world_size
        args.test_drop_number = 0
        per_rank_batch_size = None if args.iterable_dataset == 'custom' else batch_size//world_size
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=per_rank_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )

    sampler = torch.utils.data.SequentialSampler(dataset)

    drop_last = False # COMMENT: this is already solved by the complex logic of last_shape and drop_number.

    # the GPUs in the same model parallel group receive the same data
    if distributed: # TODO reformat this, but it is not urgent
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        batch_sampler = DistributedBatchSampler(sampler,
                                                batch_size,
                                                drop_last,
                                                rank,
                                                world_size,
                                                gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      batch_size,
                                                      drop_last)
    last_len = len(dataset) % batch_size
    batch_per_worker = batch_size // world_size
    last_shape = [batch_per_worker] * (last_len//batch_per_worker) # some processes get full batch
    if last_len != 0:
        if last_len % batch_per_worker != 0:
            last_shape.append(last_len % batch_per_worker) # one process get the rest (<1 batch)
        drop_number = world_size - ((last_len-1)//batch_per_worker + 1)
        # other processes get nothing, but append 1 for running. will drop later according to drop_number.
        for j in range(drop_number): 
            last_shape.append(1)
    else:
        drop_number = 0
    if split=='val':
        args.val_last_shape = last_shape
        args.val_drop_number = drop_number
    elif split=='test':
        args.test_last_shape = last_shape
        args.test_drop_number = drop_number
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                                              )
    return data_loader


def make_dataset_full(path, split, args, create_dataset_function, 
        dataset_weights=None, random_mapping=True, is_train_data=False, batch_from_same_dataset=False, **kwargs):
    """function to create datasets+tokenizers for common options"""
    print_all('make dataset ' + str(path), level='DEBUG')
    assert isinstance(path, list)

    if (is_train_data and args.iterable_dataset) or (not is_train_data and args.iterable_dataset_eval): # cannot indexed
        # the random mapping is flexible and efficient, but sometimes we have pratical issue
        # For instance, someone just gives you a iterable dataset, e.g. webdataset
        from .webds import ConfiguredResampledShards, DataPipeline
        valid_types = (ConfiguredResampledShards, DataPipeline)
        
        assert split[0] == 1, 'Iterable dataset cannot auto split.'
        ds = []
        for p in path:
            d = create_dataset_function(p, args)
            assert isinstance(d, valid_types)
            ds.append(d)
        # ds = ChainDataset(ds) # please merge them in a url if chain
        if batch_from_same_dataset:
            assert args.num_workers <= 1, 'We cannot control the actual speed of different workers, may mix different iterable parts.'
        ds = AlterDataset(ds, weights=dataset_weights, seed=args.seed, batch_from_same_dataset=batch_from_same_dataset, batch_size=args.batch_size)
        return ds

    if split is None:
        split = [1.] 
    if not should_split(split):
        ds = []
        for p in path:
            d = create_dataset_function(p, args)
            ds.append(d)
        ds = ConcatDataset(ds, weights=dataset_weights)
        if random_mapping:
            if args.epochs is not None: # not auto-scale, but use a given number of epoches.
                ds = RandomDataset(ds, scale=args.epochs, seed=args.seed)
            else:
                world_size = torch.distributed.get_world_size(
                    group=mpu.get_data_parallel_group())
                if is_train_data:
                # only train-dataset will set this to True,
                # so we enlarge it to make sure that the data is sufficient.
                    scale = max(200, 1 + (args.train_iters * args.batch_size * args.gradient_accumulation_steps * world_size) // len(ds))
                else:
                    scale = max(200, 1 + ((1 + args.train_iters // args.eval_interval) * args.eval_iters * args.eval_batch_size * args.gradient_accumulation_steps * world_size) // len(ds))
                ds = RandomMappingDataset(ds, scale=scale)
        return ds 
    else:
        # must first split datasets, then reweight/concat, finally random-mapping.
        # this order avoids overlapping.
        train_ds, valid_ds, test_ds = [], [], []
        for p in path:
            d = create_dataset_function(p, args)
            if should_split(split):
                dtrain, dvalid, dtest = split_ds(d, split, block_size=args.block_size, seed=args.seed)
                train_ds.append(dtrain)
                valid_ds.append(dvalid)
                test_ds.append(dtest)
        train_ds = ConcatDataset(train_ds, weights=dataset_weights)
        valid_ds = ConcatDataset(valid_ds, weights=dataset_weights)
        test_ds = ConcatDataset(test_ds, weights=dataset_weights)
        if random_mapping:
            world_size = torch.distributed.get_world_size(
                group=mpu.get_data_parallel_group())
            scale = max(200, 1 + (args.train_iters * args.batch_size * world_size) // len(train_ds))
            train_ds = RandomMappingDataset(train_ds, scale=scale)
            scale = max(200, 1 + ((1 + args.train_iters // args.eval_interval) * args.eval_iters * args.eval_batch_size * args.gradient_accumulation_steps * world_size) // len(valid_ds))
            valid_ds = RandomMappingDataset(valid_ds, scale=scale)
            test_ds = RandomMappingDataset(test_ds)
        return train_ds, valid_ds, test_ds

def make_loaders(args, create_dataset_function, collate_fn=None):
    """makes training/val/test
    Args:
        args.train_data, args.valid_data, args.test_data: str. Paths to the dataset.
        args.split: str. format: "8,1,1". how to split train_data.
        args.dataset_type: use to create the right datasets. 
    """
    make_dataset = partial(make_dataset_full, 
                        create_dataset_function=create_dataset_function, batch_from_same_dataset=args.batch_from_same_dataset)

    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    
    split = get_split(args)

    data_set_args = {
        'path': args.train_data,
        'split': split,
    }

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]
    
    # make datasets splits and tokenizer
    train = None
    valid = None
    test = None

    if args.train_data is not None:
        train = make_dataset(**data_set_args, args=args, dataset_weights=args.train_data_weights, is_train_data=True)
        if should_split(split):
            train, valid, test = train

    # make training and val dataset if necessary
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid = make_dataset(**eval_set_args, args=args, random_mapping=not args.strict_eval)
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test = make_dataset(**eval_set_args, args=args, random_mapping=not args.strict_eval)

    # wrap datasets with data loader
    if train is not None and args.batch_size > 0:
        train = make_data_loader(train, batch_size, args, split='train', collate_fn=collate_fn)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
    if valid is not None:
        valid = make_data_loader(valid, eval_batch_size, args, split='val', collate_fn=collate_fn)
        args.do_valid = True
    else:
        args.do_valid = False
    if test is not None:
        test = make_data_loader(test, eval_batch_size, args, split='test', collate_fn=collate_fn)
        args.do_test = True
    else:
        args.do_test = False

    return train, valid, test



def get_split(args):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1-split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s/final_sum for s in splits]

def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0]) 
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split) / sum(split) != 1.

def split_ds(ds, split=[.8,.2,.0], block_size = 10000, seed=131):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum

    assert block_size <= len(ds)

    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.array(range(block_size)))
    for i, f in enumerate(split):
        if f != 0:
            proportion = block_size*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            rtn_ds[i] = BlockedRandomSplitDataset(ds, indices[range(start_idx, start_idx+max(split_, 1))], block_size)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds

class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:  
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence, weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * weights[i])
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, weights=None, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sizes = self.cumsum(self.datasets, self.weights)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]

class RandomMappingDataset(data.Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, **kwargs):
        self.wrapped_data = ds
        self.scale = scale

    def __len__(self):
        return len(self.wrapped_data) * self.scale

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])
        index = rng.randint(len(self.wrapped_data))
        return self.wrapped_data[index]

class RandomDataset(data.Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    The indices are pre-processed.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, seed=131, **kwargs):
        self.wrapped_data = ds
        self.scale = scale
        self.indices = np.random.default_rng(seed).permutation(np.array(range(len(ds))))

    def __len__(self):
        return len(self.wrapped_data) * self.scale

    def __getitem__(self, index):
        return self.wrapped_data[int(self.indices[index % len(self.wrapped_data)])]

class BlockedRandomSplitDataset(data.Dataset):
    '''
    Dataset wrapper to access a subset of another dataset.
    Use block algorithm to reduce memory.
    In each block, using the `indices` items.
    '''
    def __init__(self, ds, indices, block_size, **kwargs):
        if type(indices) is not np.ndarray:
            indices = np.array(indices)
        indices = np.sort(indices)
        self.block_size = block_size
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.indices = indices
        self.len = len(indices) * (len(ds) // block_size) + np.sum(indices < (len(ds) % block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wrapped_data[(index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)]]

class AlterDataset(IterableDataset):
    def __init__(self, datasets, weights=None, seed=0, batch_from_same_dataset=False, batch_size=1):
        super().__init__()
        self.seed = seed
        self.datasets = datasets
        self.batch_from_same_dataset = batch_from_same_dataset
        self.batch_size = batch_size # only used when batch_from_same_dataset is True
        if weights is None:
            self.weights = [1. /  len(self.datasets)] * len(self.datasets)
        else:
            s = sum(weights)
            self.weights = [w / s for w in weights]
    
    def __iter__(self):
        iterators = [iter(d) for d in self.datasets]
        # Assume that all datasets iterate follow the seed (mprank, seed, dataloader-worker-id(0 if not used in dataloader)), auto-detect at iter()
        try:
            from sat.mpu import get_data_parallel_rank
            dp_rank = get_data_parallel_rank()
        except Exception:
            dp_rank = 0
        if self.batch_from_same_dataset:
            rng = np.random.default_rng(seed=[self.seed])
        else:
            rng = np.random.default_rng(seed=[dp_rank, self.seed])

        # sampling according to weights from streaming data
        while True:
            index = rng.choice(len(iterators), p=self.weights)
            # if stop iteration, remove the iterator
            try:
                if self.batch_from_same_dataset:
                    # we need to make sure the consecutive batch_size samples are from the same iterable dataset.
                    # but accumulate grad does not work.
                    for i in range(self.batch_size - 1):
                        yield next(iterators[index])
                yield next(iterators[index])
            except StopIteration:
                del iterators[index]
                del self.weights[index]
                if len(iterators) == 0:
                    break
                s = sum(self.weights)
                self.weights = [w / s for w in self.weights]
                from sat.helpers import print_rank0
                print_rank0(f'AlterDataset: remove a dataset, {len(iterators)} left.')