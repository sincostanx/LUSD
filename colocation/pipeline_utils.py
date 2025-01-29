import numpy as np
from PIL import Image
from functools import partial
import torch

"""
IO stuff
"""
def load_image(image_path: str, left=0, right=0, top=0, bottom=0, size=(512, 512)):
    image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize(size))
    return image

load_512 = partial(load_image, size=(512, 512))

"""
algo stuff
"""

@torch.no_grad()
def get_text_embeddings(pipe, text: str, device):
    tokens = pipe.tokenizer(
        [text],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]

def init_pipe(device, dtype, unet, scheduler):
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas

@torch.no_grad()
def decode(latent, pipe, im_cat = None, raw = False):
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    if raw:
        return image
    return Image.fromarray(image)