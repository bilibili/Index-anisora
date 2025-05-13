'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        if vit=='base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]     
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
        elif vit=='large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder,default_cfgs['vit_large_patch16_224_in21k'])      
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('./weights/bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 
        
        text_width = 768
        self.img_proj = nn.Linear(text_width, text_width)   
        self.img_key_proj = nn.Linear(text_width, text_width)
        self.head_diff = nn.Linear(2*text_width, 1)
    
    def inference(self, image, caption, image_key):
        image_embeds_list = []
        for i in range(image.shape[1]):#image torch.Size([30, 10, 3, 224, 224])
            img = image[:, i]
            image_embeds = self.visual_encoder(img) #torch.Size([75, 3, 224, 224]) ==> torch.Size([75, 197, 768])
            image_embeds_list.append(image_embeds.unsqueeze(0))
        image_embeds_all = torch.mean(torch.cat(image_embeds_list, 0), 0)
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,  return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')


        fea_1 = image_embeds_all[:,0]
        fea_2 = text_output.last_hidden_state[:,0]
        fea_1 = self.img_proj(fea_1)
        fea_2 = self.img_key_proj(fea_2)
        sim = self.head_diff(torch.cat([fea_1, fea_2], -1))
        sim =  F.sigmoid(sim)[:, 0]
        return sim
        
    def forward(self, image, caption, score, image_key, alpha):
        image_embeds_list = []
        for i in range(image.shape[1]):#image torch.Size([30, 10, 3, 224, 224])
            img = image[:, i]
            image_embeds = self.visual_encoder(img) #torch.Size([75, 3, 224, 224]) ==> torch.Size([75, 197, 768])
            image_embeds_list.append(image_embeds.unsqueeze(0))
        image_embeds_all = torch.mean(torch.cat(image_embeds_list, 0), 0)
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
        fea_1 = image_embeds_all[:,0]
        fea_2 = text_output.last_hidden_state[:,0]
        fea_1 = self.img_proj(fea_1)
        fea_2 = self.img_key_proj(fea_2)
        sim = self.head_diff(torch.cat([fea_1, fea_2], -1))
        sim =  F.sigmoid(sim)[:, 0]
        loss_diff = (torch.tensor(score).to('cuda') - sim)
        print('sim:', sim, 'score', score)
        loss_diff = torch.mean(torch.abs(loss_diff))
        return loss_diff, loss_diff, loss_diff

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


def blip_pretrain(pretrained='', **kwargs):
    model = BLIP_Pretrain(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print(msg.missing_keys)
        # assert(len(msg.missing_keys)==0)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  
