import torch

mochi_latents_mean = torch.tensor(
    [
        -0.06730895953510081,
        -0.038011381506090416,
        -0.07477820912866141,
        -0.05565264470995561,
        0.012767231469026969,
        -0.04703542746246419,
        0.043896967884726704,
        -0.09346305707025976,
        -0.09918314763016893,
        -0.008729793427399178,
        -0.011931556316503654,
        -0.0321993391887285,
    ]
).view(1, 12, 1, 1, 1)
mochi_latents_std = torch.tensor(
    [
        0.9263795028493863,
        0.9248894543193766,
        0.9393059390890617,
        0.959253732819592,
        0.8244560132752793,
        0.917259975397747,
        0.9294154431013696,
        1.3720942357788521,
        0.881393668867029,
        0.9168315692124348,
        0.9185249279345552,
        0.9274757570805041,
    ]
).view(1, 12, 1, 1, 1)
mochi_scaling_factor = 1.0


def normalize_dit_input(model_type, latents, latent_pkg = None):
    if model_type == "mochi":
        latents_mean = mochi_latents_mean.to(latents.device, latents.dtype)
        latents_std = mochi_latents_std.to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents, None
    elif model_type == "hunyuan":
        return latents * 0.476986, None
    elif model_type == "cog15B":
        # Mocking data.
        tmp_dev = latents.device
        # vae_mock = torch.rand(1, 33, 64, 48, 80, device=tmp_dev)
        # vae_mock = torch.rand(1, 25, 64, 60, 80, device=tmp_dev)
        # vae_mock = torch.rand(1, 3, 64, 90, 160, device=tmp_dev)
        # vae_mock = torch.full((1, 25, 64, 16, 24), fill_value=0.5, dtype=torch.bfloat16, device=tmp_dev)
        # vae_mock = torch.full((1, 31, 64, 64, 120), fill_value=0.5, dtype=torch.bfloat16, device=tmp_dev)
        vae_mock = torch.full((1, 11, 64, 48, 80), fill_value=0.5, dtype=torch.bfloat16, device=tmp_dev)
        encoder_hidden_states_mock = torch.rand(1, 256, 4096, device=tmp_dev)
        vector_mock = torch.rand(1, 1152, device=tmp_dev)

        latents = vae_mock
        encoder_hidden_states = encoder_hidden_states_mock
        vector = vector_mock
        return latents, {"crossattn": encoder_hidden_states, "vector": vector}
    elif model_type == "wan":
        # latent_pkg is made up with (latents, vae1, clip, t5)
        # if latent_pkg != None:
        #     pass

        latents, vae1, clip, t5 = latent_pkg
        tmp_dev = latents.device
        # mock it.
        # mock_latent_size = [1, 16, 13, 60, 104]
        # latents = torch.rand(*mock_latent_size, device=tmp_dev)
        # vae1 = torch.rand(*mock_latent_size, device=tmp_dev)
        # clip = torch.rand(1, 257, 1280, device=tmp_dev)
        # t5 = torch.rand(1, 149, 4096, device=tmp_dev)
        # post-process
        lat_h, lat_w = vae1[0].shape[2:]
        msk = torch.ones(1, vae1[0].shape[1]*4-3, lat_h, lat_w, device=tmp_dev)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]    #torch.Size([4, 13, 16, 24])

        vae1todo = torch.cat([msk, vae1[0]], dim=0)
        noise = torch.randn_like(latents)
        t5 = torch.cat([t5, t5.new_zeros(1, 512 - t5.size(1), t5.size(2))], dim=1)
        return latents, {"vae1todo": vae1todo, "t5": t5, "clip": clip, "noise": noise}
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")
