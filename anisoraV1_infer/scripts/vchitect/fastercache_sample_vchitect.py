import torch
from fastercache.models.vchitect.pipeline import VchitectXLPipeline
import random
import numpy as np
import os


from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
# from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from fastercache.models.vchitect.attention import Attention, VchitectAttnProcessor

logger = logging.get_logger(__name__)

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

def fastercache_tsb_forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor, freqs_cis: torch.Tensor, full_seqlen: int, Frame: int, counter=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        if counter>30 and counter%2==0:
            attn_output, context_attn_output = self.cached_attn_output, self.cached_context_attn_output
        else:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
                freqs_cis=freqs_cis,
                full_seqlen=full_seqlen,
                Frame=Frame,
            )
            self.cached_attn_output, self.cached_context_attn_output = attn_output, context_attn_output

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def infer(args):
    pipe = VchitectXLPipeline(args.ckpt_path)

    for _name, _module in pipe.transformer.named_modules():
        if _module.__class__.__name__=='JointTransformerBlock':
            _module.__class__.forward  = fastercache_tsb_forward

    with open(args.test_file,'r') as f:
        for lines in f.readlines():
            for seed in range(1):
                set_seed(seed)
                prompt = lines.strip('\n')
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    video = pipe.acc_call(
                        prompt,
                        negative_prompt="",
                        num_inference_steps=100,
                        guidance_scale=7.5,
                        width=768,
                        height=432, #480x288  624x352 432x240 768x432
                        frames=40
                    )

                images = video

                from fastercache.models.vchitect.utils import save_as_mp4
                import sys,os
                duration = 1000 / 8

                save_dir = args.save_dir
                os.makedirs(save_dir,exist_ok=True)

                save_as_mp4(images, os.path.join(save_dir, prompt)+'-0.mp4', duration=duration)
                
import sys,os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_known_args()[0]
    infer(args)

if __name__ == "__main__":
    main()
