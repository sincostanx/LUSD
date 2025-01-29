from typing import Tuple, Union, Optional, List

import torch
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm.auto import tqdm
import os

from colocation.pipeline_utils import get_text_embeddings, denormalize, init_pipe, decode
import torch.nn.functional as F
from colocation.attn_processor import aggregate_attention, get_all_attention

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

def merge_attention(cur_cross_attention, cur_self_attention, index, self_attn_power=1):
    self_map = cur_self_attention.reshape(32**2, 32**2)
    self_map = torch.matrix_power(self_map, self_attn_power)

    c_map = cur_cross_attention[0, index]
    if c_map.ndim > 2:
        c_map = c_map.mean(axis=0)
    
    c_map = self_map @ c_map.reshape(32**2, 1)
    c_map = c_map.reshape(32, 32)

    return c_map

class DDSLoss:
 
    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, get_raw=False,
                           guidance_scale=7.5):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t_original = e_t
            if get_raw:
                return e_t_uncond, e_t
            if guidance_scale < 0:
                e_t = e_t - e_t_uncond
            else:
                e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
            
        if get_raw:
            return e_t
        
        pred_z0_t = (z_t - sigma_t * e_t_original) / alpha_t
        pred_z0_uncond = (z_t - sigma_t * e_t_uncond) / alpha_t
        info = {
            'et_std': torch.std(e_t_original),
            'eu_std': torch.std(e_t_uncond),
            'pred_z0_t': pred_z0_t,
            'pred_z0_u': pred_z0_uncond
        }
        return e_t, pred_z0_t, info
        
    def get_csd_loss(self, z_source: T, z: T, text_embeddings: T, eps: TN = None, mask=None, t=None,
                 timestep: Optional[int] = None, 
                     reg_scale=0,
                     direction_scale=None, mask_direction=False, current_threshold=100, cmap_scale=0,
                    ) -> TS:
        
        with torch.inference_mode():
            z_t, eps_, timestep_, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, pred_z0, info = self.get_eps_prediction(z_t, timestep_, text_embeddings, alpha_t, sigma_t, guidance_scale=-1)

            # regularization
            grad_std = torch.std(e_t).item()
            c_map = torch.ones((64, 64), device=z.device) # default uniform mask

            if mask_direction and (grad_std >= current_threshold):
                # get current cross-attn A^{t,e}_C
                cur_cross_attention = (
                    get_all_attention(self.controller, is_cross=True, device="cuda")
                    .float()
                    .detach()
                )
                cur_cross_attention = F.interpolate(cur_cross_attention.permute(0, 3, 1, 2), (32, 32), mode="bicubic")

                # get current self-attn A^t_S
                cur_self_attention = (
                    get_all_attention(self.controller, is_cross=False, device="cuda")
                    .float()
                    .detach()
                )

                # refined cross-attn \hat{A}^t_C
                c_map = merge_attention(cur_cross_attention, cur_self_attention, self.target_index, 1)
                c_map = F.interpolate(c_map.unsqueeze(0).unsqueeze(0), (64, 64), mode="bicubic").squeeze()
                    
                # normalize to M
                vmin = c_map.min()
                vmax = c_map.max()
                c_map = (c_map - vmin) / (vmax - vmin)
                c_map = torch.clamp(c_map, 0, 1)

                # moving average (alpha)
                if self.moving_avg_cmap is not None:
                    c_map = (1 - self.cross_attn_momentum) * self.moving_avg_cmap + self.cross_attn_momentum * c_map
                
                self.moving_avg_cmap = c_map
                
            # progressive narrowing (beta)
            c_map = c_map * (cmap_scale) + 1 * (1-cmap_scale)
            direction = (1-reg_scale) * c_map * e_t + reg_scale * (z-z_source)

            # gradient normalization
            if (direction_scale is not None) and (not torch.all(direction == 0)):
                direction = direction / torch.std(direction) * direction_scale

            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * direction 
            
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        
        csd_loss = grad_z.clone() * z
        del grad_z
        
        info = {'timestep': timestep_.item(),
                'min': torch.min(e_t).item(),
                'max': torch.max(e_t).item(),
                'std': grad_std if (not torch.all(direction == 0)) else 1e9,
                'dmin': torch.min(direction).item(),
                'dmax': torch.max(direction).item(),
                'dstd': torch.std(direction).item(),
                'cmap_min': torch.min(c_map).item(),
                'cmap_max': torch.max(c_map).item(),
                'grad_z': direction.detach().cpu().numpy(),
                'cmap': c_map.detach().cpu().numpy(),
            **info}
        
        return csd_loss.sum() / (z.shape[2] * z.shape[3]), log_loss, pred_z0, z_t, info
        
    def __init__(self, device, pipe: StableDiffusionPipeline, controller, dtype=torch.float32, t_min=50, t_max=950, cross_attn_momentum=0):
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.pipe = pipe
        self.controller = controller
        self.prediction_type = pipe.scheduler.prediction_type
        self.cross_attn_momentum = cross_attn_momentum
        self.moving_avg_cmap = None
        print("prediction_type", self.prediction_type)

def image_optimization(
        pipeline: StableDiffusionPipeline, controller, image: np.ndarray, text_source: str, text_target: str, 
        num_iters=300, reg_scale=2e-2, learning_rate=1, t_min=50, t_max=950, mask=None,
        device="cuda",
        ds_high=0.15,
        ds_low=0.01,
        speed=1,
        threshold=1e-2,
        threshold_decay_rate=0.99,
        cross_attn_momentum=0.1,
        outdir=None,
        save_step=1,
        target_index=None,
        use_direction_scale=True,
    ) -> None:
    
    #################################
    # sanity check: save config
    data = {
        "num_iters": num_iters,
        "reg_scale": reg_scale,
        "learning_rate": learning_rate,
        "t_min": t_min,
        "t_max": t_max,
        "ds_high": ds_high,
        "ds_low": ds_low,
        "speed": speed,
        "threshold": threshold,
        "threshold_decay_rate": threshold_decay_rate,
        "cross_attn_momentum": cross_attn_momentum,
        "use_direction_scale": use_direction_scale,
    }
    import json
    save_path = os.path.join(outdir, ".config.json")
    os.makedirs(outdir, exist_ok=True)
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    #################################

    assert outdir is not None
    assert target_index is not None

    dds_loss = DDSLoss(device, pipeline, controller, t_min=t_min, t_max=t_max, cross_attn_momentum=cross_attn_momentum)
    dds_loss.target_index = target_index
    enable_mask_direction = len(dds_loss.target_index) > 0
    print(f"target_index = {target_index}")
    print(f"Enable mask direction: {enable_mask_direction}")

    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
        embedding_text = get_text_embeddings(pipeline, text_source, device=device)
        embedding_text_target = get_text_embeddings(pipeline, text_target, device=device)

    z_target = z_source.clone()
    z_target.requires_grad = True
    
    optimizer = SGD(params=[z_target], lr=learning_rate)
    
    ds_scaler = 1
    ds_high *= ds_scaler
    ds_low *= ds_scaler
    
    sigmoid = lambda z: 1/(1 + np.exp(-z))
    direction_schedule = np.linspace(-speed, speed, num_iters)
    direction_schedule = sigmoid(direction_schedule) 
    direction_schedule = (1-direction_schedule) * (ds_high - ds_low) + ds_low

    cmap_schedule = np.linspace(0, 1, num_iters)
    
    grad_counts = []
    for i in tqdm(range(num_iters)):
        
        counter = 0
        current_threshold = threshold
        while True:
            loss, log_loss, _, _, info = dds_loss.get_csd_loss(
                z_source, z_target, torch.stack([embedding_text, embedding_text_target], dim=1), 
                reg_scale=reg_scale, 
                mask=mask,
                direction_scale = direction_schedule[i] if use_direction_scale else None,    
                timestep=None,
                mask_direction=enable_mask_direction if i > 0 else False,
                current_threshold=current_threshold,
                cmap_scale=cmap_schedule[i],
            )
            
            counter +=1
            if info['std'] >= current_threshold:
                break
            current_threshold *= threshold_decay_rate
            dds_loss.controller.ignore_step()
            
        # print(f"counter: {counter}")
        grad_counts.append(counter - 1)

        optimizer.zero_grad()
        (2000 * loss).backward()
        optimizer.step()
        dds_loss.controller.update_attn_state()
        
        if (i % save_step == 0) or (i == (num_iters - 1)):
            out = decode(z_target, pipeline, im_cat=image)
            
            save_path = os.path.join(outdir, f"{i:06d}.png")
            os.makedirs(outdir, exist_ok=True)
            out.save(save_path)

    out = decode(z_target, pipeline, im_cat=None)
    return out