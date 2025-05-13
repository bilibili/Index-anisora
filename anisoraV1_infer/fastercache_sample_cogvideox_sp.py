import traceback,copy

import torch
import numpy as np
import random
def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
from videosys import VideoSysEngine
import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from fastercache.models.cogvideox.diffusion_video import SATVideoDiffusionEngine
from fastercache.models.cogvideox.arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from fastercache.utils.utils import init_process_groups, all_to_all, sp_split, sp_gather
from fastercache.models.cogvideox.sgm.util import initialize_context_parallel
import torch.distributed as dist

from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image
import pdb

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            # yield l.strip()
            cnt += 1
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        with imageio.get_writer(save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


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



import math
import copy
import torch
import torch.nn.functional as F

from sat import mpu
from sat.mpu import get_model_parallel_world_size, ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, gather_from_model_parallel_region, copy_to_model_parallel_region, checkpoint


from sat.mpu.utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from sat.ops.layernorm import LayerNorm

from sat.transformer_defaults import HOOKS_DEFAULT, standard_attention, split_tensor_along_last_dim

from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.mpu.layers import ColumnParallelLinear
from fastercache.models.cogvideox.sgm.util import instantiate_from_config

from fastercache.models.cogvideox.sgm.modules.diffusionmodules.openaimodel import Timestep
from fastercache.models.cogvideox.sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
from sat.ops.layernorm import LayerNorm, RMSNorm


def sp_attention_forward(layer, hidden_states, mask, **kw_args):
    layer = layer.transformer.layers[kw_args['layer_id']].attention
    attention_fn = attention_fn_default
    if 'attention_fn' in layer.hooks:
        attention_fn = layer.hooks['attention_fn']

    mixed_raw_layer = layer.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, layer.stride)

    dropout_fn = layer.attention_dropout if layer.training else None

    query_layer = layer._transpose_for_scores(mixed_query_layer)
    key_layer = layer._transpose_for_scores(mixed_key_layer)
    value_layer = layer._transpose_for_scores(mixed_value_layer)

    # rotary position embedding 
    if layer.transformer.is_rotary_emb:
        query_layer, key_layer = layer.transformer.position_embeddings(
            query_layer, key_layer, kw_args['position_ids'],max_seqlen=kw_args['position_ids'].max()+1,
            layer_id=kw_args['layer_id']
        )
    # rank = dist.get_rank()
    world_size = dist.get_world_size()
    query_layer = all_to_all(query_layer, 2, 1)
    key_layer = all_to_all(key_layer, 2, 1)
    value_layer = all_to_all(value_layer, 2, 1)

    if world_size >= 4:
        query_layer=query_layer[:,:,:-2]
        key_layer=key_layer[:,:,:-2]
        value_layer=value_layer[:,:,:-2]

    context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

    if world_size >= 4:
        context_layer=F.pad(context_layer,(0,0,0,2))

    context_layer = all_to_all(context_layer, 1, 2)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (layer.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = layer.dense(context_layer)

    if layer.training:
        output = layer.output_dropout(output)
    return output


def fastercache_transformer_forward(self, input_ids, position_ids, attention_mask, *,
                output_hidden_states=False, **kw_args):
    
    if True:
        assert len(input_ids.shape) >= 2
        batch_size, query_length = input_ids.shape[:2]

        if attention_mask is None:
            # Definition: None means full attention
            attention_mask = torch.ones(1, 1, device=input_ids.device)
        elif isinstance(attention_mask, int) and (attention_mask < 0):
            # Definition: -1 means lower triangular attention mask
            attention_mask = torch.ones(query_length, query_length, 
                                        device=input_ids.device).tril()
            
        attention_mask = attention_mask.type_as(
                next(self.parameters())
            )
        assert len(attention_mask.shape) == 2 or \
               len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}


        ## Image embedding
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:  # default
            hidden_states = HOOKS_DEFAULT['word_embedding_forward'](self, input_ids, output_cross_layer=output_cross_layer,**kw_args)

        # handle position embedding
        # Position embedding
        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == hidden_states.shape[1], (position_ids.shape, hidden_states.shape)
            position_embeddings = HOOKS_DEFAULT['position_embedding_forward'](self, position_ids, output_cross_layer=output_cross_layer, **kw_args)
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        world_size = dist.get_world_size()
        if(world_size>=4):
            hidden_states=F.pad(hidden_states,(0,0,0,2))#BTC,padT

        hidden_states = sp_split(hidden_states)
        output_per_layers = []
        if self.checkpoint_activations:
            # define custom_forward for checkpointing
            def custom(start, end, kw_args_index, cross_layer_index):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]

                    # recover kw_args and output_cross_layer
                    flat_inputs = inputs[2:]
                    kw_args, output_cross_layer = {}, {}
                    for k, idx in kw_args_index.items():
                        kw_args[k] = flat_inputs[idx]
                    for k, idx in cross_layer_index.items():
                        output_cross_layer[k] = flat_inputs[idx]
                    # -----------------

                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        output_this_layer_obj, output_cross_layer_obj = {}, {}
                        if 'layer_forward' in self.hooks:
                            layer_ret = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, position_ids=position_ids, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        else:
                            layer_ret = layer(
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, position_ids=position_ids, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        if isinstance(layer_ret, tuple):
                            layer_ret = layer_ret[0] # for legacy API
                        x_, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj
                        if output_hidden_states:
                            output_this_layer['hidden_states'] = x_
                        output_per_layers_part.append(output_this_layer)

                    # flatten for re-aggregate keywords outputs
                    flat_outputs = []
                    for output_this_layer in output_per_layers_part:
                        for k in output_this_layer:
                            # TODO add warning for depth>=2 grad tensors
                            flat_outputs.append(output_this_layer[k])
                            output_this_layer[k] = len(flat_outputs) - 1
                    for k in output_cross_layer:
                        flat_outputs.append(output_cross_layer[k])
                        output_cross_layer[k] = len(flat_outputs) - 1
                    # --------------------

                    return (x_, output_per_layers_part, output_cross_layer, *flat_outputs)
                return custom_forward

            # prevent to lose requires_grad in checkpointing.
            # To save memory when only finetuning the final layers, don't use checkpointing.
            if self.training:
                hidden_states.requires_grad_(True)

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            output_this_layer = []
            while l < num_layers:
                args = [hidden_states, attention_mask]
                # flatten kw_args and output_cross_layer
                flat_inputs, kw_args_index, cross_layer_index = [], {}, {}
                for k, v in kw_args.items():
                    flat_inputs.append(v)
                    kw_args_index[k] = len(flat_inputs) - 1
                for k, v in output_cross_layer.items():
                    flat_inputs.append(v)
                    cross_layer_index[k] = len(flat_inputs) - 1
                # --------------------
                if l + self.checkpoint_skip_layers >= num_layers:
                    # no checkpointing
                    hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                    custom(l, l + chunk_length, kw_args_index, cross_layer_index)(*args, *flat_inputs)
                else:
                    hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                    checkpoint(custom(l, l + chunk_length, kw_args_index, cross_layer_index), *args, *flat_inputs)
                
                # recover output_per_layers_part, output_cross_layer
                for output_this_layer in output_per_layers_part:
                    for k in output_this_layer:
                        output_this_layer[k] = flat_outputs[output_this_layer[k]]
                for k in output_cross_layer:
                    output_cross_layer[k] = flat_outputs[output_cross_layer[k]]
                # --------------------

                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            output_this_layer = []
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]

                output_this_layer_obj, output_cross_layer_obj = {}, {}

                if 'layer_forward' in self.hooks: # customized layer_forward
                    layer_ret = self.hooks['layer_forward'](*args,
                        layer_id=torch.tensor(i),
                        **kw_args,
                        position_ids=position_ids,
                        **output_cross_layer,
                        output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj
                    )
                else:
                    layer_ret = layer(*args, layer_id=torch.tensor(i), **kw_args, position_ids=position_ids, **output_cross_layer,
                        output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj)
                if isinstance(layer_ret, tuple):
                    layer_ret = layer_ret[0] # for legacy API
                hidden_states, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj

                if output_hidden_states:
                    output_this_layer['hidden_states'] = hidden_states
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        logits = copy_to_model_parallel_region(logits)
        world_size = dist.get_world_size()
        logits = sp_gather(logits)
        if world_size>=4:
            logits=logits[:,:-2]
        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args, parallel_output=self.parallel_output)
        else:
            logits_parallel = HOOKS_DEFAULT['final_forward'](self, logits, **kw_args, parallel_output=self.parallel_output)

        outputs = [logits_parallel]
        outputs.extend(output_per_layers)
        
        return outputs



import torch.fft
@torch.no_grad()
def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft


def fastercache_dit_forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        self.counter+=1
        if self.counter >=18 and self.counter % 5 !=0:
            kwargs["seq_length"] = t * h * w // (self.patch_size**2)
            kwargs["images"] = x[:1]
            kwargs["emb"] = emb[:1]
            kwargs["encoder_outputs"] = context[:1]
            kwargs["text_length"] = context.shape[1]
            kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
            kwargs['counter'] = self.counter

            self.transformer.hooks.clear()
            self.transformer.hooks.update(self.hooks)
            single_output = self.transformer(**kwargs)[0]

            (bb, tt, cc, hh, ww) = single_output.shape
            cond = rearrange(single_output, "B T C H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            lf_c, hf_c = fft(cond.float())

            if self.counter <= 40:
                self.delta_lf = self.delta_lf * 1.1
            if self.counter >= 30:
                self.delta_hf = self.delta_hf * 1.1

            new_hf_uc = self.delta_hf + hf_c
            new_lf_uc = self.delta_lf + lf_c

            combine_uc = new_lf_uc + new_hf_uc
            combined_fft = torch.fft.ifftshift(combine_uc)
            recovered_uncond = torch.fft.ifft2(combined_fft).real
            recovered_uncond = rearrange(recovered_uncond.to(single_output.dtype), "(B T) C H W -> B T C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            output = torch.cat([single_output,recovered_uncond])
        else:
            kwargs["seq_length"] = t * h * w // (self.patch_size**2)
            kwargs["images"] = x
            kwargs["emb"] = emb
            kwargs["encoder_outputs"] = context
            kwargs["text_length"] = context.shape[1]
            kwargs['counter'] = self.counter

            kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
            self.transformer.hooks.clear()
            self.transformer.hooks.update(self.hooks)
            output = self.transformer(**kwargs)[0]
            
            if self.counter>=16: 
                (bb, tt, cc, hh, ww) = output.shape
                cond = rearrange(output[0:1].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)
                uncond = rearrange(output[1:2].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)

                lf_c, hf_c = fft(cond)
                lf_uc, hf_uc = fft(uncond)

                self.delta_lf = lf_uc - lf_c
                self.delta_hf = hf_uc - hf_c

        return output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def fastercache_layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
    if True:
        assert self.transformer.layernorm_order != "sandwich", "SP for Sandwidth layernorm order not supported currently"
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        has_text = dist.get_rank()==0
        if has_text:
            text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
            img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)
        else:
            img_hidden_states = hidden_states
        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        if has_text:
            text_attention_input = layer.input_layernorm(text_hidden_states)
            text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)
            attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        else:
            attention_input = img_attention_input

        counter = kwargs['counter']

        offload=os.environ.get("offload","0")
        if counter >= 18 and counter%3!=0 and layer.cached_attn[-1].shape[0]>=attention_input.shape[0]:
            if offload=="0":
                attention_output = layer.cached_attn[1][:attention_input.shape[0]] + (layer.cached_attn[1][:attention_input.shape[0]] - layer.cached_attn[0][:attention_input.shape[0]])*0.3
            else:
                attention_output = layer.cached_attn[1][:attention_input.shape[0]].to(attention_input.device) + (layer.cached_attn[1][:attention_input.shape[0]].to(attention_input.device) - layer.cached_attn[0][:attention_input.shape[0]].to(attention_input.device))*0.3
        else:
            attention_output = sp_attention_forward(layer, attention_input, mask, **kwargs)
            if offload!="0":
                if counter==15:
                    cpu_opt=attention_output.cpu()
                    layer.cached_attn = [cpu_opt, cpu_opt]
                elif counter>15:
                    layer.cached_attn = [layer.cached_attn[-1],attention_output.cpu()]
            else:
                if counter==15:
                    layer.cached_attn = [attention_output,attention_output]
                elif counter>15:
                    layer.cached_attn = [layer.cached_attn[-1],attention_output]

        if has_text:
            text_attention_output = attention_output[:, :text_length]  # (b,n,d)
            img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)
        else:
            img_attention_output = attention_output

        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        if has_text:
            text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)
            text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
            text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
            mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        else:
            mlp_input = img_mlp_input

        mlp_output = layer.mlp(mlp_input, **kwargs)

        if has_text:
            img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
            text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        else:
            img_mlp_output = mlp_output

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        if has_text:
            text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)
            hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)
        else:
            hidden_states = img_hidden_states
        return hidden_states

class child():
    def __init__(self,init_args):###hard code so not needed
        #pdb.set_trace()
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
            os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '33956'
        ####change to videosys and sat init_process_group
        try:
            init_process_groups()
        except:
            traceback.print_exc()
        initialize_context_parallel(1)
        assert dist.get_world_size() in [1, 2, 4, 8], "Sequence parallel size only allowed to be 1, 2, 4, 8"

        self.offload=os.environ.get("offload","0")
        py_parser = argparse.ArgumentParser(add_help=False)
        known, args_list = py_parser.parse_known_args()

        args = get_args(args_list)
        args = argparse.Namespace(**vars(args), **vars(known))
        del args.deepspeed_config
        args.model_config.first_stage_config.params.cp_size = 1
        args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
        args.model_cls = SATVideoDiffusionEngine

        if self.offload=="1":
            args.model_config.network_config.params.latent_width=136
            self.image_size=[720,1088]
        else:
            self.image_size=[720,1280]

        model_cls=args.model_cls
        if isinstance(model_cls, type):
            model = get_model(args, model_cls)
        else:
            model = model_cls

        load_checkpoint(model, args)
        model.eval()
        set_seed(554)
        self.model=model
        self.args=args

    def generate(self, extra_args):
        #pdb.set_trace()
        model=self.model
        args=copy.deepcopy(self.args)
        
        if(extra_args["num_frames"]==49):
            args.sampling_num_frames=13
        elif(extra_args["num_frames"]==97):
            args.sampling_num_frames=25

        image_size = self.image_size

        if args.image2video:
            chained_trainsforms = []
            # chained_trainsforms.append(TT.Resize(size=image_size, interpolation=1))
            chained_trainsforms.append(TT.ToTensor())
            transform = TT.Compose(chained_trainsforms)

        sample_func = model.sample
        T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
        num_samples = [1]
        force_uc_zero_embeddings = ["txt"]
        device = model.device

        import types

        for _name, _module in model.named_modules():

            if _module.__class__.__name__ == 'BaseTransformer':
                _module.__class__.forward = fastercache_transformer_forward

            if _module.__class__.__name__ == 'DiffusionTransformer':
                _module.__class__.forward = fastercache_dit_forward

            if _module.__class__.__name__ == 'AdaLNMixin':
                _module.fastercache_layer_forward = fastercache_layer_forward.__get__(_module)
                model.model.diffusion_model.hooks['layer_forward'] = _module.fastercache_layer_forward

        set_seed(extra_args["seed"])
        with torch.no_grad():
            # for text, cnt in tqdm(data_iter):
            if 1:
                # all model on GPU
                if self.offload=="0":
                    model=model.to(device)
                else:
                    model=model.to("cpu")
                    torch.cuda.empty_cache()
                    # TODO: broadcast image2video
                    model.first_stage_model=model.first_stage_model.to(device)
                if args.image2video:
                    
                    text, image_path = extra_args["prompt"],extra_args["image_path"]
                    if len(image_path)==1:   
                        image = Image.open(image_path[0]).convert('RGB')
                        image = transform(image).unsqueeze(0).to('cuda')
                        image = resize_for_rectangle_crop(image, image_size, reshape_mode='center').unsqueeze(0)
                        image = image * 2.0 - 1.0
                        image = image.unsqueeze(2).to(torch.bfloat16)
                        image = model.encode_first_stage(image, None)
                        image = image.permute(0, 2, 1, 3, 4).contiguous()
                        pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                        image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
                    if len(image_path)==2:
                        image = Image.open(image_path[0]).convert('RGB')
                        image = transform(image).unsqueeze(0).to('cuda')
                        image = resize_for_rectangle_crop(image, image_size, reshape_mode='center').unsqueeze(0)
                        image = image * 2.0 - 1.0
                        image = image.unsqueeze(2).to(torch.bfloat16)
                        image = model.encode_first_stage(image, None)
                        image = image.permute(0, 2, 1, 3, 4).contiguous()
        
                        image_mid = Image.open(image_path[1]).convert('RGB')
                        image_mid = transform(image_mid).unsqueeze(0).to('cuda')
                        image_mid = resize_for_rectangle_crop(image_mid, image_size, reshape_mode='center').unsqueeze(0)
                        image_mid = image_mid * 2.0 - 1.0
                        image_mid = image_mid.unsqueeze(2).to(torch.bfloat16)
                        image_mid = model.encode_first_stage(image_mid, None)
                        image_mid = image_mid.permute(0, 2, 1, 3, 4).contiguous()
                        pad_shape1 = (image.shape[0], T-2, C, H // F, W // F)
                        image = torch.concat([image, torch.zeros(pad_shape1).to(image.device).to(image.dtype), image_mid], dim=1)                        
                    if len(image_path)==3:
                        image = Image.open(image_path[0]).convert('RGB')
                        image = transform(image).unsqueeze(0).to('cuda')
                        image = resize_for_rectangle_crop(image, image_size, reshape_mode='center').unsqueeze(0)
                        image = image * 2.0 - 1.0
                        image = image.unsqueeze(2).to(torch.bfloat16)
                        image = model.encode_first_stage(image, None)
                        image = image.permute(0, 2, 1, 3, 4).contiguous()
                        image_last = Image.open(image_path[2]).convert('RGB')
                        image_last = transform(image_last).unsqueeze(0).to('cuda')
                        image_last = resize_for_rectangle_crop(image_last, image_size, reshape_mode='center').unsqueeze(0)
                        image_last = image_last * 2.0 - 1.0
                        image_last = image_last.unsqueeze(2).to(torch.bfloat16)
                        image_last = model.encode_first_stage(image_last, None)
                        image_last = image_last.permute(0, 2, 1, 3, 4).contiguous()
        
                        image_mid = Image.open(image_path[1]).convert('RGB')
                        image_mid = transform(image_mid).unsqueeze(0).to('cuda')
                        image_mid = resize_for_rectangle_crop(image_mid, image_size, reshape_mode='center').unsqueeze(0)
                        image_mid = image_mid * 2.0 - 1.0
                        image_mid = image_mid.unsqueeze(2).to(torch.bfloat16)
                        image_mid = model.encode_first_stage(image_mid, None)
                        image_mid = image_mid.permute(0, 2, 1, 3, 4).contiguous()

                        pad_shape1 = (image.shape[0], (T-3)//2, C, H // F, W // F)
                        pad_shape2 = (image.shape[0], T-3-((T-3)//2), C, H // F, W // F)
                        image = torch.concat([image, torch.zeros(pad_shape1).to(image.device).to(image.dtype), image_mid, torch.zeros(pad_shape2).to(image.device).to(image.dtype), image_last], dim=1)
                else:
                    image = None

                if self.offload!="0":
                    model.first_stage_model=model.first_stage_model.to("cpu")
                    torch.cuda.empty_cache()
                    model.conditioner=model.conditioner.to(device)

                value_dict = {
                    "prompt": text,
                    "negative_prompt": "",
                    "num_frames": torch.tensor(T).unsqueeze(0),
                }

                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
                )
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

                c["concat"] = image
                # uc["concat"] = torch.zeros_like(image)
                uc["concat"] = image

                if self.offload!="0":
                    model.conditioner=model.conditioner.to("cpu")
                    torch.cuda.empty_cache()
                    model.model=model.model.to(device)
                for index in range(args.batch_size):
                    # all reload model on GPU
                    if self.offload == "0":
                        model=model.to(device)
                    ###########SET COUNTER###################
                    model.model.diffusion_model.counter = 0
                    samples_z = sample_func(
                        c,
                        uc=uc,
                        batch_size=1,
                        shape=(T, C, H // F, W // F),
                    )
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                    # if self.offload == "0":
                    # Unload the model from GPU to save GPU memory
                    model = model.to("cpu")
                    torch.cuda.empty_cache()
                    model.first_stage_model=model.first_stage_model.to(device)
                    latent = 1.0 / model.scale_factor * samples_z
                    if self.offload != "0":
                        latent=latent.cpu()

                    # Decode latent serial to save GPU memory
                    recons = []
                    loop_num = (T - 1) // 2
                    for i in range(loop_num):
                        if i == 0:
                            start_frame, end_frame = 0, 3
                        else:
                            start_frame, end_frame = i * 2 + 1, i * 2 + 3
                        if i == loop_num - 1:
                            clear_fake_cp_cache = True
                        else:
                            clear_fake_cp_cache = False
                        with torch.no_grad():
                            recon = model.first_stage_model.decode(latent[:, :, start_frame:end_frame].contiguous().to(device), clear_fake_cp_cache=clear_fake_cp_cache)

                        recons.append(recon)
                    if self.offload != "0":
                        model.first_stage_model=model.first_stage_model.to("cpu")
                        torch.cuda.empty_cache()
                    recon = torch.cat(recons, dim=2).to(torch.float32)
                    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                    if mpu.get_model_parallel_rank() == 0:
                        # save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)
                        save_video_as_grid_and_mp4(samples, extra_args["output_path"], fps=args.sampling_fps)
                    dist.barrier()

class Args:
    def __init__(self,num_gpus):
        self.num_gpus=num_gpus
        self.pipeline_cls=child
        self.config=None


if __name__ == "__main__":
    engine = VideoSysEngine(Args(num_gpus=8))
    engine.generate(extra_args={})
