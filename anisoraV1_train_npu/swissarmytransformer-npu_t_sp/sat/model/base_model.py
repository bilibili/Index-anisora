# -*- encoding: utf-8 -*-
'''
@File    :   base_model.py
@Time    :   2021/10/01 22:40:33
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import math
import random
import torch
import inspect
import warnings
import argparse
from sat.model.registry import model_registry, MetaModel

from sat.model.transformer import BaseTransformer, standard_attention
from sat.arguments import update_args_with_file, overwrite_args_by_dict, set_random_seed
from sat.training.model_io import load_checkpoint
from sat.helpers import print_rank0

from sat.transformer_defaults import HOOKS_DEFAULT, ARGS_DEFAULT
from sat.resources import auto_create
from sat.mpu.initialize import get_node_rank, get_model_parallel_rank, destroy_model_parallel, initialize_model_parallel
from sat.mpu.operation import mp_split_model_rank0, mp_split_model_receive, mp_merge_model_rank0, mp_merge_model_send
from sat.arguments import reset_random_seed

def non_conflict(func):
    '''mark a hook function as non-conflict,
    so that it can be compatible with any already defined hooks.
    e.g. PrefixTuningMixin.attention_fn
    '''
    func.non_conflict = True
    return func

def replacable(func):
    '''mark a hook function as replacable,
    so that it can be replaced by mixins added after it.
    e.g. FP32AttentionMixin.attention_fn
    '''
    func.replacable = True
    return func

class BaseMixin(torch.nn.Module):
    non_conflict = non_conflict
    replacable = replacable
    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    # can define hook-functions here
    # a hook, if default or replacable, can be overrided by mixins added after it.
    # a hook can be augmented by non_conflict hooks added after it.
    # default -> 0~n replacable  -> 0~n non_conflict
    # ...

    # If the hook is just a pre- or post- transformation,
    # You can use @non_conflict to mark it,
    # and run `old_impl` to make it compatible with other mixins.
    # Eg., 
    # 
    # @non_conflict
    # def attention_fn(q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
    #     new_q, new_k, new_v = pre_hack(q, k, v)
    #     attn_result = old_impl(q, k, v, mask, dropout_fn, **kw_args)
    #     attn_result = post_hack(attn_result)
    #     return attn_result


class BaseModel(torch.nn.Module, metaclass=MetaModel):
    def __init__(self, args, transformer=None, params_dtype=torch.float, **kwargs):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
        if transformer is not None:
            self.transformer = transformer
        else:
            # check if model-only mode
            from sat.arguments import _simple_init
            success = _simple_init(model_parallel_size=args.model_parallel_size, seed=args.seed if hasattr(args, 'seed') else 1234)

            args_dict = {k: (getattr(args, v[0]) if hasattr(args, v[0]) else v[1]) for k, v in ARGS_DEFAULT.items()}
            self.transformer = BaseTransformer(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                max_sequence_length=args.max_sequence_length,
                layernorm_order=args.layernorm_order,
                **args_dict,
                hooks=self.hooks,
                params_dtype=params_dtype,
                skip_init=args.skip_init,
                device=torch.cuda.current_device() if hasattr(args, 'use_gpu_initialization') and args.use_gpu_initialization else torch.device('cpu'),
                **kwargs
            )

    def reinit(self, mixin_names=None):  # will be called when loading model, None means all
        # if some mixins are loaded, overrides this function
        for k, m in self.mixins.items():
            if mixin_names is None or k in mixin_names:
                m.reinit(self)

    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)

        self.mixins[name] = new_mixin  # will auto-register parameters
        object.__setattr__(new_mixin, 'transformer', self.transformer)  # cannot use pytorch set_attr

        self.collect_hooks_()
        if reinit:
            new_mixin.reinit(self)  # also pass current mixins

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()

    def get_mixin(self, name):
        return self.mixins[name]

    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)

    def collect_hooks_(self):
        names = list(HOOKS_DEFAULT.keys())
        hooks = {}
        hook_origins = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'

            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if hasattr(getattr(m, name), 'non_conflict'):
                        # check getattr(m, name), who must accept old_impl as an argument
                        signature = inspect.signature(getattr(m, name))
                        if 'old_impl' not in signature.parameters:
                            raise ValueError(f'Hook {name} at {mixin_name} must accept old_impl as an argument.')
                        # -------------
                        if name in hooks:
                            old_impl = hooks[name]
                        elif name == 'attention_fn': # the only hook without self
                            old_impl = HOOKS_DEFAULT[name]
                        else:
                            old_impl = partial(HOOKS_DEFAULT[name], self) # relax! `partial` does not affect the signature
                        old_origin = hook_origins.get(name, 'default')
                        hooks[name] = partial(getattr(m, name), old_impl=old_impl)
                        hook_origins[name] = mixin_name + ' -> ' + old_origin
                    elif name in hooks and not hasattr(hooks[name], 'replacable'): # if this hook name is already registered
                        raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    else: # new hook
                        if name in hooks and hasattr(hooks[name], 'replacable'):
                            warnings.warn(f'Hook {name} at {mixin_name} replaces {hook_origins[name]}.')
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name

        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def add_model_specific_args(cls, parser):
        # recorded in arguments.py: add_model_config_args
        return parser

    @classmethod
    def from_pretrained_base(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, overwrite_args={}, **kwargs):
        '''Load a pretrained checkpoint of the current model.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it. None will create a new model-only one with defaults.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: the url of the model. Default: SAT_URL.
                prefix: the prefix of the checkpoint. Default: ''.
            Returns:
                model: the loaded model.
                args: the loaded args.
        '''
        if os.path.exists(name) and os.path.isdir(name):
            model_path = name
        else:
            model_path = auto_create(name, path=home_path, url=url)
        # create a new args if not provided
        if args is None:
            args = cls.get_args()
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        specific_iteration = kwargs.pop('specific_iteration', None)
        model = get_model(args, cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, load_path=model_path, prefix=prefix, specific_iteration=specific_iteration)
        return model, args
    
    @classmethod
    def from_pretrained(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, "world size should be a multiplier of new model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.skip_init = True
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference': # For multi-node inference, we should prevent rank 0 eagerly printing some info.
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, "world size should be equal to model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args
    
    @classmethod
    def list_avail_args(cls, print=True):
        '''List all available args of the current model.'''
        parser = argparse.ArgumentParser()
        from sat.arguments import add_model_config_args
        add_model_config_args(parser)
        # add args of the current model
        if hasattr(cls, 'add_model_specific_args'):
            cls.add_model_specific_args(parser)
        if print:
            from sat.helpers import print_parser
            print_parser(parser)
        return parser

    @classmethod
    def get_args(cls, **kwargs):
        '''Get the parsed args of the current model.
            Args:
                **kwargs: will override the default args.
            Returns:
                args: the parsed args.
        '''
        parser = cls.list_avail_args(print=False)
        # use parser to parse kwargs
        args = parser.parse_args([])
        for k, v in kwargs.items():
            if hasattr(args, k) or k in ['fp16']: # non-arch args but affect building models
                setattr(args, k, v)
            else:
                print_rank0(f'warning: Unknown arg {k} for class {cls.__name__}.', level='DEBUG')
                setattr(args, k, v)
        return args

class AutoModel():
    @classmethod
    def from_pretrained_base(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, overwrite_args={}, **kwargs):
        '''Automatically find the class and instantiate it. Auto-download.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: manually specified url for the `name` model.
        '''
        if os.path.exists(name) and os.path.isdir(name):
            model_path = name
        else:
            model_path = auto_create(name, path=home_path, url=url)
        if args is None:
            args = argparse.Namespace() # null, fill later
            null_args = True
        else:
            null_args = False
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        if not hasattr(args, 'model_class'):
            raise ValueError('model_config.json must have key "model_class" for AutoModel.from_pretrained.')
        model_cls = model_registry.get(args.model_class)
        if null_args:
            # fill args with default values, if not provided
            model_default_args = model_cls.get_args()
            for k, v in model_default_args.__dict__.items():
                if not hasattr(args, k):
                    setattr(args, k, v)
        model = get_model(args, model_cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, load_path=model_path, prefix=prefix)
        return model, args
    
    @classmethod
    def from_pretrained(cls, name, args=None, *, home_path=None, url=None, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, "world size should be a multiplier of new model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.skip_init = True
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference': # For multi-node inference, we should prevent rank 0 eagerly printing some info.
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, "world size should be equal to model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(name, args=args, home_path=home_path, url=url, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args
    
def get_model(args, model_cls, **kwargs):
    """Build the model."""
    import torch
    from sat.helpers import print_rank0,print_all
    from sat import mpu

    print_rank0(f'building {model_cls.__name__} model ...')
    if 'params_dtype' not in kwargs:
        if hasattr(args, 'fp16') and args.fp16:
            params_dtype = torch.half
        elif hasattr(args, 'bf16') and args.bf16:
            params_dtype = torch.bfloat16
        else:
            params_dtype = torch.float32
    else:
        # pop params_dtype from kwargs
        params_dtype = kwargs.pop('params_dtype')

    from sat.helpers import check_if_zero3
    if check_if_zero3(args):
        import deepspeed
        with deepspeed.zero.Init():
            model = model_cls(args, params_dtype=params_dtype, **kwargs)
    else:
        model = model_cls(args, params_dtype=params_dtype, **kwargs)

    if mpu.get_data_parallel_rank() == 0:
        print_all(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)
    
    if hasattr(args, 'fp16') and args.fp16:
        model.half()
    elif hasattr(args, 'bf16') and args.bf16:
        model.bfloat16()

    try: # TODO: is this useful?
        if not hasattr(args, 'device'):
            args.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        model = model.to(args.device)
    except Exception as e:
        print_all(e)
    
    return model
