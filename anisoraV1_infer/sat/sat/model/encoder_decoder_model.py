# -*- encoding: utf-8 -*-
'''
@File    :   encoder_decoder_model.py
@Time    :   2021/11/22 23:35:28
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
from .base_model import BaseModel, BaseMixin, load_checkpoint, get_model
from sat.mpu.mappings import copy_to_model_parallel_region
from sat import update_args_with_file
from sat.resources import auto_create

class EncoderFinalMixin(BaseMixin):
    def final_forward(self, logits, **kwargs):
        logits = copy_to_model_parallel_region(logits)
        return logits


class EncoderDecoderModel(torch.nn.Module):
    def __init__(self, args, encoder=None, decoder=None, tie_word_embeddings=True, **kwargs):
        super(EncoderDecoderModel, self).__init__()
        if encoder is not None:
            assert isinstance(encoder, BaseModel)
            self.encoder = encoder
        else:
            self.encoder = BaseModel(args, **kwargs)
        self.encoder.add_mixin("final", EncoderFinalMixin())
        
        if decoder is not None:
            assert isinstance(decoder, BaseModel)
            self.decoder = decoder
        else:
            dec_args = argparse.Namespace(**vars(args))
            dec_args.enc_hidden_size = dec_args.hidden_size  # used for cross attn
            override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads', 'layernorm_order'
                              'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
            for name in override_attrs:
                dec_attr = getattr(dec_args, 'dec_' + name, None)
                if dec_attr is not None:  # else use encoder-config
                    setattr(dec_args, name, dec_attr)
            self.decoder = BaseModel(args, is_decoder=True, **kwargs)

        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            self.decoder.transformer.word_embeddings = self.encoder.transformer.word_embeddings

    def reinit(self, mixin_names): # please use different mixin names for encoder and decoder
        self.encoder.reinit(mixin_names)
        self.decoder.reinit(mixin_names)

    def disable_untrainable_params(self):
        self.encoder.disable_untrainable_params()
        self.decoder.disable_untrainable_params()

    def encode(self, input_ids, position_ids, attention_mask=None, **kw_args):
        encoder_outputs, *_dumps = self.encoder(input_ids, position_ids, attention_mask, **kw_args)
        return encoder_outputs
    
    def decode(self, input_ids, position_ids, attention_mask, encoder_outputs,cross_attention_mask=None, **kw_args):
        if attention_mask is None:
            batch_size, seq_length = input_ids.size()[:2]
            seq_ids = torch.arange(seq_length, device=input_ids.device)
            attention_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            attention_mask = attention_mask.to(self.decoder.transformer.word_embeddings.weight.dtype)
            attention_mask = attention_mask[:, None, :, :]
        # If no context, please explicitly pass ``encoder_outputs=None''
        return self.decoder(input_ids, position_ids, attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
    
    def forward(self, enc_input_ids, enc_position_ids, dec_input_ids, dec_position_ids, *, enc_attention_mask=None, dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        # Please use self.decoder for auto-regressive generation.
        batch_size, seq_length = enc_input_ids.size()[:2]
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length, dtype=self.encoder.transformer.word_embeddings.weight.dtype, device=enc_input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_position_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(dec_input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask, **kw_args)
        return (encoder_outputs, decoder_outputs, *mems)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EncoderDecoderModel', 'T5 or Bart')
        group.add_argument("--dec-num-layers", type=int, default=None)
        group.add_argument("--dec-hidden-size", type=int, default=None)
        group.add_argument("--dec-num-attention-heads", type=int, default=None)
        group.add_argument("--dec-max-sequence-length", type=int, default=None)
        group.add_argument("--dec-inner-hidden-size", type=int, default=None)
        group.add_argument("--dec-hidden-size-per-attention-head", type=int, default=None)
        group.add_argument("--dec-layernorm-order", type=str, default=None)
        return parser

    @classmethod
    def from_pretrained(cls, args, name, *, home_path=None, url=None): # TODO update model-only mode
        model_path = auto_create(name, path=home_path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(args, cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args
