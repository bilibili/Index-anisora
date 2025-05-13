import pdb

import torch
import numpy as np
from .utils import _broadcast_tensor, _extract_into_tensor



class _WrappedModel_DiT:
    def __init__(self, model, diffusion, device=None, class_emb_null=None):
        self.model = model
        self.diffusion = diffusion
        self._predict_xstart_from_eps = diffusion._predict_xstart_from_eps
        
        self.diffusion_t_map = list(diffusion.use_timesteps)
        self.diffusion_t_map.sort()
        
        self.diffusion_t = [self.diffusion_t_map[i] for i in range(diffusion.num_timesteps)] # list(range(diffusion.num_timesteps))
        self.diffusion_t = np.array(self.diffusion_t)
        
        self.diffusion_sqrt_alpha_cumprod = np.array([diffusion.sqrt_alphas_cumprod[i] for i in range(diffusion.num_timesteps)])
        self.fm_steps = [(1 - self.diffusion_sqrt_alpha_cumprod[i]**2)**0.5/(self.diffusion_sqrt_alpha_cumprod[i] + (1 - self.diffusion_sqrt_alpha_cumprod[i]**2)**0.5) for i in range(len(self.diffusion_t))]
        
        self.fm_steps = torch.tensor([0] + self.fm_steps, device=device)
        self.y_null = class_emb_null




        
    def __call__(self, x, t, y, kwargs):

        N = len(self.diffusion_t)
        B,C,H,W = x.shape
        diffusion_x = torch.zeros_like(x)
        diffusion_t = _extract_into_tensor(self.diffusion_t, t-1, t.shape).long()


        t_fm = self.fm_steps[t]
        diffusion_x_tmp = _extract_into_tensor(self.diffusion.sqrt_alphas_cumprod, t-1, x.shape) * x / ( 1 + 1e-4 - _broadcast_tensor(t_fm,x.shape)) 
        diffusion_x_tmp = diffusion_x_tmp.to(torch.float)
        diffusion_x = torch.where(_broadcast_tensor(t,x.shape) == N, x, diffusion_x_tmp)


        y_null_batch = torch.cat([self.y_null[0].unsqueeze(0)]*B, dim=0)
        y_new = torch.cat([y, y_null_batch], 0)
        

        model_output = self.model(torch.cat([diffusion_x,diffusion_x],dim=0), torch.cat([diffusion_t,diffusion_t],dim=0), y_new, **kwargs)
        model_output = model_output[:B]
        model_output, _ = torch.split(model_output, C, dim=1)
        x0_diffusion = self._predict_xstart_from_eps(x_t=diffusion_x, t=t-1, eps=model_output)
        vt = (x - x0_diffusion) / (_broadcast_tensor(t_fm,x.shape))
        vt = vt.to(diffusion_x.dtype)
        return vt



class _WrappedModel_Sora:
    def __init__(self, model, guidance_scale, y_null, timesteps, num_timesteps, mask_t):
        self.model = model
        self.guidance_scale = guidance_scale    
        self.y_null = y_null
        
        self.timesteps = [torch.tensor([0], device=model.device)] + timesteps[::-1]
        self.timesteps = torch.cat(self.timesteps, dim=0)
        self.fm_steps = [x/num_timesteps for x in self.timesteps]
        self.mask_t = mask_t
        
    def __call__(self, x, t, y, kwargs):
        y = torch.cat([y, self.y_null], dim=0)
        
        t_in = self.timesteps[t]        
        
        x_in = torch.cat([x,x], dim=0)
        # breakpoint()
        mask_t_upper = self.mask_t >= t_in.unsqueeze(1)
        kwargs["x_mask"] = mask_t_upper.repeat(2, 1)

        t_in = torch.cat([t_in,t_in], dim=0)
        with torch.no_grad():
            pred = self.model(x_in, t_in, y,  **kwargs).chunk(2, dim=1)[0]
        # breakpoint()
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        
        return -v_pred



class _WrappedModel_Wan:
    def __init__(self, model, timesteps, num_timesteps, context_null, guide_scale):
        self.model = model
        self.context_null = context_null
        self.guide_scale = guide_scale
        fm_steps = torch.cat([timesteps,torch.zeros_like(timesteps[0]).view(1)])
        self.time_steps = torch.flip(fm_steps, dims=[0])
        self.fm_steps = self.time_steps/num_timesteps
        
        
    def __call__(self, x, t, y, kwargs):
        self.time_steps = self.time_steps.to(t.device)
        t = self.time_steps[t]
        noise_pred_cond = self.model(x, t=t, context=y, **kwargs)[0]
        noise_pred_uncond = self.model(x, t=t, context=self.context_null, **kwargs)[0]
        noise_pred = noise_pred_uncond + self.guide_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred




class _WrappedModel_FLUX:
    def __init__(self, model, timesteps, num_timesteps):
        self.model = model
        fm_steps = torch.cat([timesteps,torch.zeros_like(timesteps[0]).view(1)])
        self.time_steps = torch.flip(fm_steps, dims=[0])
        self.fm_steps = self.time_steps/num_timesteps
            
        
    def __call__(self, x, t, y, kwargs):
        t = self.time_steps[t]
        t = t.expand(x.shape[0]).to(x.dtype) / 1000
        pred = self.model(hidden_states=x, timestep=t, **kwargs)[0]
        return pred
