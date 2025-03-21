import abc
import math
from typing import List

import torch
from diffusers.models.attention_processor import Attention

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, heads: int, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, heads, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        if is_cross:
            attn = self.forward(attn, heads, is_cross, place_in_unet)
        else:
            attn[h // 2 :] = self.forward(attn[h // 2 :], heads, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
        return attn

    def update_attn_state(self):
        self.between_steps()
        self.cur_step += 1

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStoreClassPrompts(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, heads, is_cross: bool, place_in_unet: str):
        if self.start <= self.cur_step <= self.end:
            if attn.shape[1] <= 64**2:  # avoid memory overhead
                spatial_res = int(math.sqrt(attn.shape[1]))
                attn_store = attn.reshape(-1, heads, spatial_res, spatial_res, attn.shape[-1])
                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                self.step_store[key].append(attn_store)
        return attn

    def between_steps(self):
        if self.start <= self.cur_step <= self.end:
            if self.attention_store is None:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] = self.attention_store[key][i] + self.step_store[key][i]

            self.latest_store = self.step_store
            self.step_store = self.get_empty_store()

    def ignore_step(self):
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        start = max(0, self.start)
        end = min(self.cur_step, self.end + 1)

        average_attention = {}
        for key in self.attention_store:
            if self.attention_store[key] is None: continue
            average_attention[key] = [item / (end - start) for item in self.attention_store[key]]
        return average_attention

    def reset(self):
        super(AttentionStoreClassPrompts, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = None

    def __init__(self, start=0, end=1000):
        super(AttentionStoreClassPrompts, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = None
        self.latest_store = None
        self.start = start
        self.end = end
        
        self.attention_masks = None

    def export(self, prefix, is_cross=True):
        if is_cross:
            for i, data in enumerate(self.attention_store['down_cross']): torch.save(data, f"{prefix}_down_cross_{i}.pt")
            for i, data in enumerate(self.attention_store['mid_cross']): torch.save(data, f"{prefix}_mid_cross_{i}.pt")
            for i, data in enumerate(self.attention_store['up_cross']): torch.save(data, f"{prefix}_up_cross_{i}.pt")
        else:
            raise NotImplementedError

class CustomAttentionStoreClassPrompts(AttentionStoreClassPrompts):
    def between_steps(self):
        if self.start <= self.cur_step <= self.end:
            if self.attention_store is None:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] = self.attention_store[key][i] + self.step_store[key][i]

            self.latest_store = self.step_store
            self.step_store = self.get_empty_store()

    def __init__(self, start=0, end=1000):
        super(AttentionStoreClassPrompts, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = None
        self.latest_store = None
        self.start = start
        self.end = end
        
        self.attention_masks = None

    def export(self, prefix, is_cross=True):
        if is_cross:
            for i, data in enumerate(self.attention_store['down_cross']): torch.save(data, f"{prefix}_down_cross_{i}.pt")
            for i, data in enumerate(self.attention_store['mid_cross']): torch.save(data, f"{prefix}_mid_cross_{i}.pt")
            for i, data in enumerate(self.attention_store['up_cross']): torch.save(data, f"{prefix}_up_cross_{i}.pt")
        else:
            raise NotImplementedError

class StoredAttnClassPromptsProcessor:
    def __init__(self, attnstore, place_in_unet, save_self_attn):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.save_self_attn = save_self_attn

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if not is_cross:
            if (attention_probs.shape[1] == 1024) and (self.save_self_attn):
                self.attnstore(attention_probs, attn.heads, False, self.place_in_unet)
        else:
            if attention_probs.shape[1] == 256:
                if self.attnstore.attention_masks is not None:
                    attention_probs = attention_probs * self.attnstore.attention_masks.view((1, -1, 1))

                cross_attention_probs = attention_probs[attn.heads * batch_size // 2 :]
                self.attnstore(cross_attention_probs.detach(), attn.heads, True, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

import torch.nn.functional as F
import numpy as np
def aggregate_attention(
    attention_store: AttentionStoreClassPrompts,
    res,
    is_cross: bool,
    from_where: List[str] = ["up", "down", "mid"],
    device="cpu",
    **kwargs,
):
    out = []
    attention_maps = attention_store.get_average_attention(**kwargs)
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if isinstance(res, int):
                if item.shape[2] == res:
                    out.append(item)
            elif isinstance(res, list):
                max_res = np.array(res).max()
                if item.shape[2] in res:
                    mod_item = F.interpolate(item, size=(max_res, max_res, 77), mode='trilinear')
                    out.append(mod_item)
            else:
                raise ValueError
    out = torch.cat(out, dim=1).mean(dim=1)

    if device == "cpu":
        return out.cpu()
    else:
        return out.to(device)

def get_all_attention(
    attention_store: AttentionStoreClassPrompts,
    device="cpu",
    is_cross=True,
):

    attention_probs = []
    skip_word = "self" if is_cross else "cross"
    for key in attention_store.latest_store:
        if attention_store.latest_store[key] is None: continue
        if skip_word in key: continue
        for item in attention_store.latest_store[key]:
            avg_head_map = item.mean(dim=1).unsqueeze(1) # average head
            attention_probs.append(avg_head_map)
        
    attention_probs = torch.cat(attention_probs, dim=1).mean(dim=1) # average layer
    if device == "cpu":
        return attention_probs.cpu()
    else:
        return attention_probs.to(device)

def register_attention_control(model, controller, processor, use_self_attn=False, **kwargs):
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = processor(attnstore=controller, place_in_unet=place_in_unet, save_self_attn=use_self_attn, **kwargs)

    model.unet.set_attn_processor(attn_procs)
    # num_layers = 5 if not use_self_attn else 10
    # controller.num_att_layers = num_layers # 5 # cross_att_count 
