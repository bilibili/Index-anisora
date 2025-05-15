import traceback

import torch
from torch.utils.data import Dataset
import json
import os
import random

class LatentDataset(Dataset):
    def __init__(
        self, json_path, num_latent_t, cfg_rate,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(512, 4096).to(torch.float32)#cog is 226,others is 256
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(512).bool()
        self.cache=None
        self.lengths = [
            # data_item["length"] if "length" in data_item else 1
            13
            for data_item in self.data_anno
        ]


    def __getitem__(self, idx):
        try:
            filename = self.data_anno[idx]["vae1"]
            vae_all=torch.load(self.data_anno[idx]["vae_all"],map_location="cpu",weights_only=True)
            vae1=torch.load(self.data_anno[idx]["vae1"],map_location="cpu",weights_only=True)
            clip=torch.load(self.data_anno[idx]["clip"],map_location="cpu",weights_only=True)[0]
            t5=torch.load(self.data_anno[idx]["t5"],map_location="cpu",weights_only=True)
            # print(11111,vae_all.shape,vae1.shape,clip.shape,t5.shape)#11111 torch.Size([16, 13, 60, 104]) torch.Size([16, 13, 60, 104]) torch.Size([257, 1280]) torch.Size([103, 4096
            assert vae_all.numel()==1297920
            assert vae1.numel()==1297920
            assert clip.numel()==328960

            self.cache=( vae_all, vae1, clip,t5)
            return vae_all, vae1, clip,t5
        except Exception as e:
            print(filename,"load fail",e)
            # traceback.print_exc()
            if type(self.cache)!=type(None):
                return self.cache
            else:
                return self.__getitem__(random.randint(0,self.__len__()))

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function0(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    latents, prompt_embeds, prompt_attention_masks = zip(*batch)
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
            ),
        )
        for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1] :, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2] :, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3] :] = 0

    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latents, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks

def latent_collate_function(batch):
    vae_all, vae1, clip, t5 = zip(*batch)

    vae_all = torch.stack(vae_all, dim=0)
    vae1 = torch.stack(vae1, dim=0)
    clip = torch.stack(clip, dim=0)
    t5 = torch.stack(t5, dim=0)####bs1 can stack, bs>1 can't because different t length
    return vae_all, vae1, clip, t5


if __name__ == "__main__":
    dataset = LatentDataset("data/Mochi-Synthetic-Data/merge.txt", num_latent_t=28)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=latent_collate_function
    )
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask in dataloader:
        print(
            latent.shape,
            prompt_embed.shape,
            latent_attn_mask.shape,
            prompt_attention_mask.shape,
        )
        import pdb

        pdb.set_trace()
