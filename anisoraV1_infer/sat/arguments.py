# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""argparser configuration"""

import argparse
import os
import torch
import json
import random
import numpy as np
import warnings
from sat import mpu

import logging
from sat.helpers import print_all, print_rank0

_GLOBAL_RANDOM_SEED = None

def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--num-layers', type=int, default=6,
                       help='num of layers')
    group.add_argument('--hidden-size', type=int, default=1024,
                       help='transformer hidden size')
    group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    group.add_argument('--vocab-size', type=int, default=100,
                       help='vocab size for tokenization. the size of word_embeddings.')
    group.add_argument('--max-sequence-length', type=int, default=512,
                       help='maximum number of position embeddings to use')
    
    # --------------- Optional hyper-parameters --------------- 
    # for developers:
    #   if you want to add more optional arch hyper-parameters, please also add them in model_io.py, base_model.py, and of course, transformer.py and anywhere in use.

    group.add_argument('--layernorm-order', type=str, default='pre',
                       help='choose from "pre", "post", "sandwich".',
                       choices=['post', # In the original Transformer.
                                'pre', # Used by most current frameworks.
                                'sandwich' # More stable.
                                ])
    # The inner-hidden-size in MLP, default "None" means 4*hidden-size
    group.add_argument('--inner-hidden-size', type=int, default=None,
                       help='inner hidden size for transformer FFN, None means 4*hidden-size')
    # The hidden-size-per-attention-head in Self and Cross Attention, 
    # default "None" means hidden-size/num-attention-heads.
    group.add_argument('--hidden-size-per-attention-head', type=int, default=None)
    # TODO: fully test it, support the generation.
    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel. only use if you are an expert.')

    group.add_argument('--skip-init', action='store_true',
                       help='skip model initialization')
    
    group.add_argument('--use-gpu-initialization', action='store_true', 
                       help='initialize model on gpu')

    group.add_argument('--num-multi-query-heads', type=int, default=0, 
                       help='use multi-query attention, num of kv groups. 0 means multi-head attention.')
    group.add_argument('--is-gated-mlp', action='store_true',
                       help='use gated MLP (GLU), common in LLAMA etc.')
    group.add_argument('--is-rotary-emb', action='store_true',
                       help='use rotary embedding, common in LLAMA etc.')
    
    # ---------------  Inessential hyper-parameters --------------- 

    # Dropout and eps hyper-parameters
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--drop-path', type=float, default=0., help='drop path probability')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--parallel-output', action='store_true', help='whether to gather model parallel outputs at final output. Need to be True if using mpu.vocab_parallel_cross_entropy.')
    
    return parser



def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--experiment-name', type=str, default="MyModel",
                       help="The experiment name for summary and checkpoint."
                       "Will load the previous name if mode==pretrain and with --load ")
    group.add_argument('--train-iters', type=int, default=None,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--batch-size', type=int, default=4,
                       help='batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--mode', type=str,
                       default='pretrain',
                       choices=['pretrain', # from_scratch / load ckpt for continue pretraining.
                                'finetune', # finetuning, auto-warmup 100 iters, new exp name.
                                'inference' # don't train.
                                ],
                       help='what type of task to use, will influence auto-warmup, exp name, iteration')
    group.add_argument('--seed', type=int, default=1234, help='random seed')
    group.add_argument('--zero-stage', type=int, default=0, choices=[0, 1, 2, 3], 
                        help='deepspeed ZeRO stage. 0 means no ZeRO.')
    group.add_argument('--sp-size', type=int, default=1, help='sp size for diffusion-transformer')

    # ---------------  Optional hyper-parameters --------------- 

    # Efficiency.
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                            'with larger models and sequences. become slow (< 1.5x), save CUDA memory.')
    # Inessential
    group.add_argument('--checkpoint-num-layers', type=int, default=1, 
                       help='chunk size (number of layers) for checkpointing. ')
    group.add_argument('--checkpoint-skip-layers', type=int, default=0,
                       help='skip the last N layers for checkpointing.')
    
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bf16 mode')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='run optimizer after every gradient-accumulation-steps backwards.')

    group.add_argument('--profiling', type=int, default=-1,
                       help='profiling, -1 means no profiling, otherwise means warmup args.profiling iters then profiling.')
    group.add_argument('--epochs', type=int, default=None,
                       help='number of train epochs')
    group.add_argument('--log-interval', type=int, default=50,
                       help='report interval')
    group.add_argument('--summary-dir', type=str, default="", help="The directory to store the summary")
    group.add_argument('--save-args', action='store_true',
                       help='save args corresponding to the experiment-name')

    # Learning rate & weight decay.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--force-train', action='store_true',
                       help='Force training even with missing keys.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--resume-dataloader', action='store_true',
                       help='Resume the dataloader when resuming training. ') 

    # distributed training related, don't use them.
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'training. One of [gloo, nccl]')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    # exit, for testing the first period of a long training
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')

    group.add_argument('--wandb', action="store_true", help='whether to use wandb')
    group.add_argument('--wandb-project-name', type=str, default="default_project",
                       help="The project name in wandb.")
    
    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                            'Defaults to `--batch-size`')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='number of iterations to run for evaluation'
                            'validation/test for')
    group.add_argument('--eval-interval', type=int, default=None,
                       help='interval between running evaluation on validation set')
    group.add_argument('--strict-eval', action='store_true',
                       help='won\'t enlarge or randomly map eval-data, and eval full eval-data.')
    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    # Training datasets.
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated filenames or corpora names for training.'
                       'Use hf://path/to/dataset to load huggingface datasets.')
    group.add_argument('--train-data-weights', nargs='+', default=None, type=int,
                        help='scaling factors for different train-data, must the same number of'
                        '--train-data or None(==1).')
    parser.add_argument('--iterable-dataset', 
                        action='store', default=False, const=True, nargs='?', choices=[True, False, 'custom'], help='custom means Dataloader batch_size=None, and you need to generate batches by yourself. This is useful for dynamic batchsize.')
    parser.add_argument('--iterable-dataset-eval', default='',
                            type=str, choices=['True', 'False', ''], help='Whether eval is iterable.')

    group.add_argument('--batch-from-same-dataset', action='store_true',
                       help='batch from the same dataset, not random. ONLY affect len(train-data-weights) > 1 and iterable dataset')

    # Validation and Test dataset.
    group.add_argument('--valid-data', nargs='*', default=None,
                       help="""Filename for validation data.""")
    group.add_argument('--test-data', nargs='*', default=None,
                       help="""Filename for testing""")
    group.add_argument('--split', default='1000,1,1',
                       help='comma-separated list of proportions for training,'
                            ' validation, and test split. '
                            'Only take effects when no valid/test specified.')

    # Efficiency.
    group.add_argument('--num-workers', type=int, default=1,
                       help="""Number of workers to use for dataloading""")
    # Sometimes, num-workders > 1 and cpu_offload (zero-stage 2) will make the validation dataloader hang.
    # I have no idea on the reason, and just set the default num-workers to 1.
    group.add_argument('--block-size', type=int, default=10000,
                       help="""Size of block to reduce memory in dataset, ignore it for most users.""")
    group.add_argument('--prefetch-factor', type=int, default=4, 
                       help='prefetching number of batches.')
    return parser


def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--length-penalty", type=float, default=0.0)
    group.add_argument("--no-repeat-ngram-size", type=int, default=0)
    group.add_argument("--min-tgt-length", type=int, default=0)
    group.add_argument("--out-seq-length", type=int, default=256)
    group.add_argument('--input-source', type=str, default='interactive',
                       help='what input mode to use, interactive or path')
    group.add_argument('--output-path', type=str, default='./samples',
                       help='path to place the generated samples')
    group.add_argument('--with-id', action='store_true',
                       help='If each line is prepended with an id.')
    group.add_argument('--max-inference-batch-size', type=int, default=12)
    group.add_argument('--device', type=int, default=-1)
    return parser


    
def add_tokenization_args(parser):
    """tokenization arguments."""

    group = parser.add_argument_group('Tokenization', 'tokenization configurations')
    group.add_argument('--tokenizer-type', type=str, default='fake', help='type name of tokenizer')
    
    # group.add_argument('--img-tokenizer-path', type=str, default=None,
    #                    help='The checkpoint file path of image tokenizer.')
    return parser

def _adjust_vocab_size(args):
    before = args.vocab_size
    after = before
    multiple = args.make_vocab_size_divisible_by
    # you should control args to let it divided by 
    # mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print_rank0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

def _simple_init(model_parallel_size=1, seed=0):
    '''Necessary initialization for torch.distributed for model-only mode'''
    args = argparse.Namespace(
        distributed_backend='nccl' if torch.distributed.is_nccl_available() and torch.cuda.is_available() else 'gloo',
        model_parallel_size=model_parallel_size,
    )
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.local_rank = int(os.getenv("LOCAL_RANK", '0')) # torchrun
    args.device = args.local_rank
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.deepspeed = False
    print_rank0("Set torch.backends.cudnn.deterministic with False.")
    set_random_seed(seed, False)
        
    if initialize_distributed(args): # first time init model parallel, print warning
        print_rank0(
                  'You are using model-only mode.\n\
For torch.distributed users or loading model parallel models, set environment variables RANK, WORLD_SIZE and LOCAL_RANK.'
                  )
        return True
    return False

def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='sat')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)
    parser = add_tokenization_args(parser)
    parser = add_text_generate_args(parser)

    # Include DeepSpeed configuration arguments
    import deepspeed
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args_list)
    if not args.iterable_dataset_eval:
        args.iterable_dataset_eval = args.iterable_dataset
    else:
        args.iterable_dataset_eval = eval(args.iterable_dataset_eval)

    if not args.train_data:
        print_rank0('No training data specified', level='WARNING')

    assert (args.train_iters is None) or (args.epochs is None), 'only one of train_iters and epochs should be set.'
    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000 # default 10k iters
        print_rank0('No train_iters (recommended) or epochs specified, use default 10k iters.', level='WARNING')

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", '0')) # torchrun
   
    if args.device == -1: # not set manually
        if torch.cuda.device_count() == 0:
            args.device = 'cpu'
        elif args.local_rank is not None:
            args.device = args.local_rank
        else:
            args.device = args.rank % torch.cuda.device_count()
            
    # local rank should be consistent with device in DeepSpeed
    if args.local_rank != args.device and args.mode != 'inference':
        raise ValueError(
            'LOCAL_RANK (default 0) and args.device inconsistent. '
            'This can only happens in inference mode. '
            'Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. '
            )

    # args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print_rank0('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))
    if args.vocab_size > 0:
        _adjust_vocab_size(args)
    
    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data), f'{args.train_data_weights} vs {args.train_data}'
    
    if args.mode != 'inference': # training with deepspeed
        args.deepspeed = True
        if args.deepspeed_config is None: # not specified
            args.deepspeed_config = os.path.join(os.path.dirname(__file__), 'training', f'deepspeed_zero{args.zero_stage}.json')
            override_deepspeed_config = True
        else:
            override_deepspeed_config = False

    assert not (args.fp16 and args.bf16), 'cannot specify both fp16 and bf16.'

    if args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank0('Automatically set fp16=True to use ZeRO.')     
        args.fp16 = True
        args.bf16 = False

    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            with open(args.deepspeed_config) as file:
                deepspeed_config = json.load(file)
            
        if override_deepspeed_config: # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else: # override args with values in deepspeed_config
            if args.rank == 0:
                print_rank0('Will override arguments with manually specified deepspeed_config!')
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config
    
    # if args.sandwich_ln: # removed in v0.3
    #     args.layernorm_order = 'sandwich'
    
    # initialize distributed and random seed because it always seems to be necessary.
    initialize_distributed(args)
    set_random_seed(args.seed)
    return args


def update_args_with_file(args, path):
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # expand relative path
    folder = os.path.dirname(path)
    for k in config:
        # all the relative paths in config are based on the folder
        if k.endswith('_path'): 
            config[k] = os.path.join(folder, config[k])
            print_rank0(f'> parsing relative path {k} in model_config as {config[k]}.')
    args = vars(args)
    for k in list(args.keys()):
        if k in config:
            del args[k]
    args = argparse.Namespace(**config, **args)
    return args


def overwrite_args_by_dict(args, overwrite_args={}):
    if 'decoder_freq' in overwrite_args:
        decoder_freq = overwrite_args['decoder_freq']
        del overwrite_args['decoder_freq']
    else:
        decoder_freq = None
    for k in overwrite_args:
        setattr(args, k, overwrite_args[k])
    if decoder_freq is not None:
        args.is_decoder = []
        for i in range(args.num_layers):
            if i % decoder_freq == 0:
                args.is_decoder.append(True)
            else:
                args.is_decoder.append(False)
    return args


def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError('model_parallel_size is inconsistent with prior configuration.'
                                 'We currently do not support changing model_parallel_size.')
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn('model_parallel_size > 1 but torch.distributed is not initialized via SAT.'
                            'Please carefully make sure the correctness on your own.')
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == 'cpu':
        pass
    else:
        torch.cuda.set_device(args.device)
    # Call the init process
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    
    if args.world_size == 1:
        from sat.helpers import get_free_port
        default_master_port = str(get_free_port())
    else:
        default_master_port = '6000'
    args.master_port = os.getenv('MASTER_PORT', default_master_port)
    init_method += args.master_ip + ':' + args.master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)
    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed: 
        import deepspeed
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank, init_method=init_method)
        # It seems that it has no negative influence to configure it even without using checkpointing.  
        deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout. 
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import _CUDA_RNG_STATE_TRACKER, _MODEL_PARALLEL_RNG_TRACKER_NAME
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1) # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0
            print_rank0(str(e), level="DEBUG")


    return True

def set_random_seed(seed, torch_deterministic = True):
    """Set random seed for reproducability."""
    if seed is not None:
        global _GLOBAL_RANDOM_SEED
        _GLOBAL_RANDOM_SEED = seed
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = torch_deterministic
        if False == torch_deterministic:
            torch.use_deterministic_algorithms(torch_deterministic)
        torch.backends.cudnn.enabled = True # False
        torch.backends.cuda.matmul.allow_tf32 = False # if set it to True will be much faster but not accurate
        try:
            import deepspeed
            if deepspeed.checkpointing.is_configured():
                mpu.model_parallel_cuda_manual_seed(seed)
        except ImportError:
            pass

def reset_random_seed(scale=1):
    assert _GLOBAL_RANDOM_SEED is not None, "You have not set random seed. No need to reset it."
    set_random_seed(_GLOBAL_RANDOM_SEED*scale)