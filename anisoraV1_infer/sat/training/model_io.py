# -*- encoding: utf-8 -*-
'''
@File    :   model_io.py
@Time    :   2021/10/05 18:39:55
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import numpy as np
import json
import argparse
import warnings
from collections import OrderedDict, namedtuple
from sat import mpu
from sat.helpers import print_rank0, print_all
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Set, Tuple, Union
import itertools

# for overriding pytorch's save/load state_dict for zero3
_EXTRA_STATE_KEY_SUFFIX = '_extra_state'  # pytorch default name


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, 'latest')


def extract_model_specific_args_from_model(args, model):
    parser = argparse.ArgumentParser()

    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, torch.nn.Module):
        for md in model.modules():  # search
            if hasattr(md, 'add_model_specific_args'):
                try:
                    md.add_model_specific_args(parser)
                except argparse.ArgumentError as e:
                    print(e)
    ret = {}
    for k in vars(parser.parse_args([])).keys():
        if hasattr(args, k):
            ret[k] = getattr(args, k)
    return ret


def extract_model_specific_args_to_dump(args, model):
    module = model.module if hasattr(model, 'module') else model
    # model_class
    to_dump = {'model_class': type(module).__name__}
    # tokenizer_type
    if hasattr(args, 'tokenizer_type') and args.tokenizer_type != 'fake':
        to_dump['tokenizer_type'] = args.tokenizer_type
        # architecture related args
    arch_args_list = ['num_layers', 'hidden_size', 'num_attention_heads', 'vocab_size',
                      'layernorm_order', 'model_parallel_size', 'max_sequence_length',
                      ]
    for name in arch_args_list:
        if hasattr(args, name) and getattr(args, name) is not None:
            to_dump[name] = getattr(args, name)

    # optional architecture related args, only save if not default
    # optional means might be changed when loading
    optional_arch_args_list = [
        ('is_decoder', False),
        ('cross_attn_hidden_size', None),
        ('use_bias', True),
        ('use_qkv_bias', False),
        ('inner_hidden_size', None),
        ('hidden_size_per_attention_head', None),
        ('cross_hidden_size_per_attention_head', None),
        ('use_final_layernorm', True),
        ('layernorm_epsilon', 1e-5),
        ('num_multi_query_heads', 0),
        ('cross_num_multi_query_heads', 0),
        ('row_parallel_linear_final_bias', True),
        ('is_gated_mlp', False),
        ('is_rotary_emb', False),
        ('parallel_output', False),
        ('num_experts', 1),
    ]
    if hasattr(module, 'transformer'):
        for name, default in optional_arch_args_list:
            if module.transformer.__dict__[name] != default:
                to_dump[name] = module.transformer.__dict__[name]

    # model specific args
    model_specific_args = extract_model_specific_args_from_model(args, module)
    to_dump.update(model_specific_args)
    return to_dump


def update_ema_parameters_to_model(optimizer):
    """update ema parameters"""
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.utils import all_gather_dp_groups
    from packaging import version
    for i, (bit16_partitions, fp32_partition) in enumerate(
            zip(optimizer.parallel_partitioned_bit16_groups, optimizer.single_partition_of_fp32_groups)):
        ema_optimizer = optimizer.optimizer
        state = ema_optimizer.state[fp32_partition]
        partition_id = dist.get_rank(group=optimizer.real_dp_process_group[i])
        bit16_partitions[partition_id].data.copy_(state['shadow'].data)
    if version.parse(deepspeed.version) >= version.parse("0.12.4"):
        all_gather_dp_groups(groups_flat=optimizer.bit16_groups_flat,
                             partitioned_param_groups=optimizer.parallel_partitioned_bit16_groups,
                             dp_process_group=optimizer.real_dp_process_group,
                             start_alignment_factor=optimizer.nccl_start_alignment_factor,
                             allgather_bucket_size=optimizer.allgather_bucket_size)
    else:
        all_gather_dp_groups(partitioned_param_groups=optimizer.parallel_partitioned_bit16_groups,
                             dp_process_group=optimizer.real_dp_process_group,
                             start_alignment_factor=optimizer.nccl_start_alignment_factor,
                             allgather_bucket_size=optimizer.allgather_bucket_size)


def restore_ema_parameters_back(optimizer):
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.utils import all_gather_dp_groups
    from packaging import version
    for i, (bit16_partitions, fp32_partition) in enumerate(
            zip(optimizer.parallel_partitioned_bit16_groups, optimizer.single_partition_of_fp32_groups)):
        partition_id = dist.get_rank(group=optimizer.real_dp_process_group[i])
        bit16_partitions[partition_id].data.copy_(fp32_partition.data)
    if version.parse(deepspeed.version) >= version.parse("0.12.4"):
        all_gather_dp_groups(groups_flat=optimizer.bit16_groups_flat,
                             partitioned_param_groups=optimizer.parallel_partitioned_bit16_groups,
                             dp_process_group=optimizer.real_dp_process_group,
                             start_alignment_factor=optimizer.nccl_start_alignment_factor,
                             allgather_bucket_size=optimizer.allgather_bucket_size)
    else:
        all_gather_dp_groups(partitioned_param_groups=optimizer.parallel_partitioned_bit16_groups,
                             dp_process_group=optimizer.real_dp_process_group,
                             start_alignment_factor=optimizer.nccl_start_alignment_factor,
                             allgather_bucket_size=optimizer.allgather_bucket_size)


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    if hasattr(args, 'deepspeed') and args.deepspeed:
        if mpu.get_data_parallel_rank() == 0:
            print_rank0('Saving Model...')
            save_ds_checkpoint(iteration, model, lr_scheduler, args)
        if optimizer is not None and optimizer.optimizer.__class__.__name__ == "FusedEmaAdam":
            update_ema_parameters_to_model(optimizer)
            if mpu.get_data_parallel_rank() == 0:
                print_rank0('Saving Ema Model...')
                save_ds_checkpoint(iteration, model, lr_scheduler, args, True)
            restore_ema_parameters_back(optimizer)

    elif args.mode == 'inference':
        os.makedirs(os.path.join(args.save, str(iteration)), exist_ok=True)
        if torch.distributed.get_rank() < args.model_parallel_size:
            torch.save({'module': model.state_dict()}, os.path.join(args.save, str(iteration), 'mp_rank_{:02d}_model_states.pt'.format(torch.distributed.get_rank())))
    else:
        raise ValueError("training without deepspeed is not supported.")
    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
        # save model_config.json for from_pretrained().
        with open(os.path.join(args.save, 'model_config.json'), 'w') as f:
            to_dump = extract_model_specific_args_to_dump(args, model)
            json.dump(to_dump, f, indent=4)

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, lr_scheduler, args, use_ema=False):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
    if not use_ema:
        save_ds_checkpoint_no_optim(model, args.save, str(iteration), client_state=sd)
    else:
        save_ds_checkpoint_no_optim(model, args.save, str(iteration) + '-ema', client_state=sd)


def save_ds_checkpoint_no_optim(model, save_dir, tag=None, client_state={}, save_latest=True):
    os.makedirs(save_dir, exist_ok=True)
    # Ensure tag is a string
    tag = str(tag)
    # Real save via deepspeed
    model._create_checkpoint_file(save_dir, tag, False)
    if model.zero_optimization_partition_weights():
        # zero3, deepspeed originally save a full model by layerwise gather, which is slow and need extra memory. Moreover, the loading also needs a lot of CPU memory (infeasible for very large model).
        torch.save({'module': zero3_state_dict(model)}, os.path.join(save_dir, tag, 'z3_rank_{:02d}_model_states.pt'.format(torch.distributed.get_rank())))
    else:
        model._save_checkpoint(save_dir, tag, client_state=client_state)
    # Save latest checkpoint tag
    if save_latest:
        with open(os.path.join(save_dir, 'latest'), 'w') as fd:
            fd.write(tag)

    return True


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank0('could not find the metadata file {} '.format(
            tracker_filename))
        raise ValueError('could not find the metadata file {}, please check --load'.format(
            tracker_filename))
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release, True


def load_checkpoint(model, args, load_path=None, prefix='', specific_iteration=None):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load

    # If model-only mode, set necessary args.
    if not hasattr(args, 'mode'):
        from copy import deepcopy
        args = deepcopy(args)
        args.mode = 'inference'

    iteration, release, success = get_checkpoint_iteration(load_path)
    if specific_iteration is not None:
        assert type(specific_iteration) == int and specific_iteration > 0
        print_rank0('Overriding checkpoint iteration to {}'.format(specific_iteration))
        iteration = specific_iteration

    if not success:
        return 0

    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    # if mpu.get_data_parallel_rank() == 0:
    print_all('global rank {} is loading checkpoint {}'.format(
        torch.distributed.get_rank(), checkpoint_name))

    # load state_dict into CPU
    sd = torch.load(checkpoint_name, map_location='cpu')

    # if given `prefix`, load a speficic prefix in the checkpoint, e.g. encoder
    new_sd = {'module': {}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    sd = new_sd

    if hasattr(model, 'module'):
        module = model.module
    else:  # inference without deepspeed
        module = model

    # only load module, other hyperparameters are just for recording.
    try:
        del sd['module']["model.diffusion_model.mixins.pos_embed.freqs_sin"]
    except:
        pass
    try:
        del sd['module']["model.diffusion_model.mixins.pos_embed.freqs_cos"]
    except:
        pass

    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)
    if len(unexpected_keys) > 0:
        print_rank0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        if args.mode == 'inference':
            if 'force_inference' in args and args.force_inference:
                print_rank0(f'Warning: Missing keys for inference: {missing_keys}.')
            else:
                raise ValueError(f'Missing keys for inference: {missing_keys}.\nIf you still want to inference anyway, pass --force_inference to args.')
        else:  # new params
            if not args.force_train:
                assert all(name.find('mixins') >= 0 or name.find('cross_attention') >= 0 for name in missing_keys), missing_keys
                assert args.mode == 'finetune'
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                if key_name.find('mixins') < 0:
                    continue
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins') + 1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names)  # initialize mixins

    # Do not need this any more, because we create optimizer after load now.
    # if args.mode != 'inference' and args.deepspeed and args.fp16:
    #     model.optimizer.refresh_fp32_params() # restore fp32 weights from module

    # Iterations.
    if args.mode == 'finetune':
        iteration = 0
    elif args.mode == 'pretrain' and not args.no_load_rng:  # rng states.
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank0('Unable to load optimizer from checkpoint {}, exiting. '
                        'Specify --no-load-rng or --finetune to prevent '
                        'attempting to load the random '
                        'state.'.format(checkpoint_name))
            exit()
    elif args.mode == 'inference':
        module.eval()

    if mpu.get_data_parallel_rank() == 0:
        print_all('> successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration


def zero3_save_to_state_dict(module, destination, prefix, keep_vars):
    r"""Save module state to the `destination` dictionary.

    The `destination` dictionary will contain the state
    of the module, but not its descendants. This is called on every
    submodule in :meth:`~torch.nn.Module.state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Args:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    for name, param in module._parameters.items():
        if param is not None:
            if hasattr(param, 'ds_id'):
                # deepspeed param for zero3
                destination[prefix + name] = param.ds_tensor.detach()
                destination[prefix + name].ds_shape = param.ds_shape  # record the unpartitioned shape
            else:
                destination[prefix + name] = param if keep_vars else param.detach()
    # zero3 did not partition registered buffers I think, so keep it.
    for name, buf in module._buffers.items():
        if buf is not None and name not in module._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
        destination[extra_state_key] = module.get_extra_state()


def zero3_state_dict(module, *args, destination=None, prefix='', keep_vars=False):
    r"""Return a dictionary containing references to the whole state of a already paritioned module.
    """
    # TODO: Remove `args` and the parsing logic when BC allows.
    if len(args) > 0:
        if destination is None:
            destination = args[0]
        if len(args) > 1 and prefix == '':
            prefix = args[1]
        if len(args) > 2 and keep_vars is False:
            keep_vars = args[2]
        # DeprecationWarning is ignored by default
        warnings.warn(
            "Positional args are being deprecated, use kwargs instead. Refer to "
            "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
            " for details.")

    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()

    local_metadata = dict(version=module._version)
    if hasattr(destination, "_metadata"):
        destination._metadata[prefix[:-1]] = local_metadata

    for hook in module._state_dict_pre_hooks.values():
        hook(module, prefix, keep_vars)
    zero3_save_to_state_dict(module, destination, prefix, keep_vars)
    for name, module in module._modules.items():
        if module is not None:
            zero3_state_dict(module, destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def _zero3_load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
    r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

    This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.
    Additionally, :attr:`local_metadata` can also contain the key
    `assign_to_params_buffers` that indicates whether keys should be
    assigned their corresponding tensor in the state_dict.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not torch.overrides.is_tensor_like(input_param):
                error_msgs.append(f'While copying the parameter named "{key}", '
                                  'expected torch.Tensor or Tensor-like object from checkpoint but '
                                  f'received {type(input_param)}'
                                  )
                continue

            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, input_param.shape, param.shape))
                continue

            if param.is_meta and not input_param.is_meta and not assign_to_params_buffers:
                warnings.warn(f'for {key}: copying from a non-meta parameter in the checkpoint to a meta '
                              'parameter in the current model, which is a no-op. (Did you mean to '
                              'pass `assign=True` to assign items in the state dictionary to their '
                              'corresponding key in the module instead of copying them in place?)')

            try:
                with torch.no_grad():
                    if assign_to_params_buffers:
                        # Shape checks are already done above
                        if (isinstance(param, torch.nn.Parameter) and
                                not isinstance(input_param, torch.nn.Parameter)):
                            setattr(self, name, torch.nn.Parameter(input_param))
                        else:
                            setattr(self, name, input_param)
                    else:
                        if hasattr(param, 'ds_id'):
                            # deepspeed param for zero3
                            param.ds_tensor.copy_(input_param)
                        else:
                            param.copy_(input_param)
            except Exception as ex:
                error_msgs.append(f'While copying the parameter named "{key}", '
                                  f'whose dimensions in the model are {param.size()} and '
                                  f'whose dimensions in the checkpoint are {input_param.size()}, '
                                  f'an exception occurred : {ex.args}.'
                                  )
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(self.__class__, "set_extra_state", torch.nn.Module.set_extra_state) is not torch.nn.Module.set_extra_state:
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)


def zero3_load_state_dict(self, state_dict: Mapping[str, Any],
                          strict: bool = True, assign: bool = False):
    r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

    If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict`.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): whether to assign items in the state
            dictionary to their corresponding keys in the module instead
            of copying them inplace into the module's current parameters and buffers.
            When ``False``, the properties of the tensors in the current
            module are preserved while when ``True``, the properties of the
            Tensors in the state dict are preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata['assign_to_params_buffers'] = assign
        _zero3_load_from_state_dict(module,
                                    local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)  # noqa: F821

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join(f'"{k}"' for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)
