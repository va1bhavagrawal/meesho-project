import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union, Dict 

import torch
import torch.nn.functional as F
from torch import nn

import sys 

import diffusers 
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention 

import numpy as np 
import cv2 
from PIL import Image 

DEBUG_ATTN = False  

class AttendExciteAttnProcessor:
    def __init__(self, name):
        super().__init__()
        self.name = name 


    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # print(f"input to attn for {self.layer_name} is of shape: {hidden_states.shape}")
        batch_size, sequence_length, _ = hidden_states.shape
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if type(encoder_hidden_states) == dict: 
            actual_encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"] 
            used_attention_maps = encoder_hidden_states["attn_assignments"] 
        else: 
            actual_encoder_hidden_states = encoder_hidden_states 
        key = attn.to_k(actual_encoder_hidden_states)
        value = attn.to_v(actual_encoder_hidden_states) 
        if type(encoder_hidden_states) == dict: 
            # assert len(encoder_hidden_states["attn_assignments"]) == len(encoder_hidden_states["encoder_hidden_states"]) 
            B = len(encoder_hidden_states["attn_assignments"]) 
            for batch_idx in range(B): 
                for idx1, idx2 in used_attention_maps[batch_idx].items(): 
                    key[batch_idx][idx1] = key[batch_idx][idx2].detach()  

        print(f"before {key.shape = }") 
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        print(f"after {key.shape = }") 

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # print(f"{encoder_hidden_states.shape = }")
        # print(f"{hidden_states.shape = }")
        # print(f"{query.shape = }")
        # print(f"{key.shape = }")
        # print(f"{attention_probs.shape = }")
        # sys.exit(0) 

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttnProcessor2_0_edited:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attn_store = None 
        self.name = name 


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Dict,  
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )
        if encoder_hidden_states is None: 
            batch_size, sequence_length, _ = hidden_states.shape 
        elif type(encoder_hidden_states) == dict: 
            batch_size, sequence_length, _ = encoder_hidden_states["encoder_hidden_states"].shape 
            # print(f"{batch_size = }")
        else: 
            batch_size, sequence_length, _ = encoder_hidden_states.shape 

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        if type(encoder_hidden_states) == dict: 
            actual_encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"] 
            attn_assignments = encoder_hidden_states["attn_assignments"] 
        else: 
            actual_encoder_hidden_states = encoder_hidden_states 

        key = attn.to_k(actual_encoder_hidden_states)
        value = attn.to_v(actual_encoder_hidden_states)

        if type(encoder_hidden_states) == dict: 
            assert len(actual_encoder_hidden_states) == len(attn_assignments) 
            for batch_idx in range(len(actual_encoder_hidden_states)): 
                for idx1, idx2 in attn_assignments[batch_idx].items(): 
                    key[batch_idx][idx1] = key[batch_idx][idx2]  

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.attn_store is None: 
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            if DEBUG_ATTN and type(encoder_hidden_states) == dict: 
                attn_map = query @ torch.transpose(key, -2, -1) 
                attn_map = F.softmax(attn_map, dim=-1) 
                # print(f"{attn_map.shape = }") 
                # print(f"{len(actual_encoder_hidden_states) = }") 
                for batch_idx in range(len(actual_encoder_hidden_states)): 
                    # print(f"{batch_idx = }") 
                    for idx1, idx2 in attn_assignments[batch_idx].items(): 
                        assert torch.allclose(attn_map[batch_idx, ..., idx1], attn_map[batch_idx, ..., idx2]), f"{attn_map.shape = }, {batch_idx = }, {idx1 = }, {idx2 = }"    

        else: 
            # attn_map = F.softmax(query @ torch.transpose(key, -1, -2), dim=-1)  
            attn_map = query @ torch.transpose(key, -2, -1)   
            attn_map = F.softmax(attn_map, dim=-1) 
            # attn_map = attn.get_attention_scores(query, key, attention_mask) 
            self.attn_store(attn_map, self.name) 
            hidden_states = attn_map @ value 

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# class AttnAddedKVProcessor2_0_edited:
#     r"""
#     Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
#     learnable key and value matrices for the text encoder.
#     """

#     def __init__(self):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError(
#                 "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
#             )

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Dict,  
#         attention_mask: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
#             deprecate("scale", "1.0.0", deprecation_message)

#         residual = hidden_states

#         hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
#         batch_size, sequence_length, _ = hidden_states.shape

#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             raise NotImplementedError(f"did not expect norm cross to be True!") 
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         if type(encoder_hidden_states) == dict: 
#             actual_encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"] 
#             attn_assignments = encoder_hidden_states["attn_assignments"] 
#         else: 
#             actual_encoder_hidden_states = encoder_hidden_states 

#         if attn.group_norm is not None: 
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)
#         query = attn.head_to_batch_dim(query, out_dim=4)

#         encoder_hidden_states_key_proj = attn.add_k_proj(actual_encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(actual_encoder_hidden_states)
#         encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
#         encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)

#         if type(encoder_hidden_states) == dict: 
#             assert len(actual_encoder_hidden_states) == len(attn_assignments) 
#             for batch_idx in range(len(actual_encoder_hidden_states)): 
#                 for idx1, idx2 in attn_assignments[batch_idx].items(): 
#                     key[batch_idx][idx1] = key[batch_idx][idx2]  

#         if not attn.only_cross_attention:
#             key = attn.to_k(hidden_states)
#             value = attn.to_v(hidden_states)
#             key = attn.head_to_batch_dim(key, out_dim=4)
#             value = attn.head_to_batch_dim(value, out_dim=4)
#             key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
#             value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
#         else:
#             key = encoder_hidden_states_key_proj
#             value = encoder_hidden_states_value_proj

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
#         hidden_states = hidden_states + residual

#         return hidden_states


# class MyAttentionProcessor: 
#     r"""
#     Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
#     learnable key and value matrices for the text encoder.
#     """

#     def __init__(self):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError(
#                 "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
#             )

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states,  
#         encoder_hidden_states,  
#         attention_mask: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         # if len(args) > 0 or kwargs.get("scale", None) is not None:
#         #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
#         #     deprecate("scale", "1.0.0", deprecation_message)

#         residual = hidden_states

#         hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
#         batch_size, sequence_length, _ = hidden_states.shape

#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             raise NotImplementedError(f"did not expected {attn.norm_cross = }") 
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         if attn.group_norm is not None: 
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         print(f"{hidden_states.shape = }") 
#         query = attn.to_q(hidden_states)
#         query = attn.head_to_batch_dim(query, out_dim=4)

#         if type(encoder_hidden_states) == dict: 
#             actual_encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"] 
#             attn_assignments = encoder_hidden_states["attn_assignments"] 
#         else: 
#             actual_encoder_hidden_states = encoder_hidden_states 

#         encoder_hidden_states_key_proj = attn.add_k_proj(actual_encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(actual_encoder_hidden_states)
#         encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
#         encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)

#         if not attn.only_cross_attention:
#             key = attn.to_k(hidden_states)
#             value = attn.to_v(hidden_states)
#             key = attn.head_to_batch_dim(key, out_dim=4)
#             value = attn.head_to_batch_dim(value, out_dim=4)
#             key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
#             value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
#         else:
#             key = encoder_hidden_states_key_proj
#             value = encoder_hidden_states_value_proj

#         if type(encoder_hidden_states) == dict: 
#             assert len(actual_encoder_hidden_states) == len(attn_assignments) 
#             for batch_idx in range(len(actual_encoder_hidden_states)): 
#                 for idx1, idx2 in attn_assignments[batch_idx].items(): 
#                     key[batch_idx][idx1] = key[batch_idx][idx2]  


#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
#         hidden_states = hidden_states + residual

#         return hidden_states


def patch_custom_attention(unet, store_attn): 
    attn_procs = {}
    cross_att_count = 0
    attn_store = None 
    if store_attn: 
        attn_store = AttentionStore(unet)  
    # print(f"printing all the unet attn_processors keys")
    for name, attn_processor in unet.attn_processors.items():
        # # print(f"{name}")
        # print(f"{attn_processor = }")
        # cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        # if name.startswith("mid_block"):
        #     hidden_size = model.unet.config.block_out_channels[-1]
        #     place_in_unet = "mid"
        # elif name.startswith("up_blocks"):
        #     block_id = int(name[len("up_blocks.")])
        #     hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
        #     place_in_unet = "up"
        # elif name.startswith("down_blocks"):
        #     block_id = int(name[len("down_blocks.")])
        #     hidden_size = model.unet.config.block_out_channels[block_id]
        #     place_in_unet = "down"
        # else:
        #     continue

        cross_att_count += 1
        # attn_procs[name] = MyAttentionProcessor()  
        # attn_procs[name] = AttnProcessor2_0_edited(name, attn_store)     
        if attn_store is None: 
            attn_procs[name] = AttnProcessor2_0_edited(name)      
        else: 
            attn_procs[name] = AttendExciteAttnProcessor(name, attn_store) 

    unet.set_attn_processor(attn_procs) 

    if store_attn: 
        return attn_store 


class AttentionStore:
    def get_empty_store(self):
        self.step_store = {}
        # cross_att_count = 0
        # for name in self.unet.attn_processors.keys():
            # cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            # if name.startswith("mid_block"):
            #     hidden_size = self.unet.config.block_out_channels[-1]
            #     place_in_unet = "mid"
            # elif name.startswith("up_blocks"):
            #     block_id = int(name[len("up_blocks.")])
            #     hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            #     place_in_unet = "up" 
            # elif name.startswith("down_blocks"):
            #     block_id = int(name[len("down_blocks.")])
            #     hidden_size = self.unet.config.block_out_channels[block_id]
            #     place_in_unet = "down" 
            # else:
            #     continue
            # self.step_store[name] = []


    def forward(self, attn, layer_name: str):
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            # if self.step_store[layer_name] != []:
                # one diffusion step has completed
                # store the current object in a pkl file
                # empty the object
                # with open(str(self.n_steps).zfill(4) + ".pkl", "wb") as f:
                #     pickle.dump(self.step_store, f)
                #     # print(f"stored step_store for step {self.n_steps}")
                # self.n_steps = self.n_steps + 1
                # self.get_empty_store()
            if layer_name in self.step_store.keys(): 
                assert self.step_store[layer_name][0].shape == attn.shape 
            else: 
                self.step_store[layer_name] = [] 
                if not self.across_timesteps: 
                    self.step_store[layer_name].append(torch.zeros_like(attn).cpu())  

            if not self.across_timesteps:  
                self.step_store[layer_name][0] = self.step_store[layer_name][0] + attn.cpu()  
            else: 
                self.step_store[layer_name].append(attn)  
        return attn 


    def __call__(self, attn, layer_name: str):
        self.forward(attn, layer_name)


    def __init__(self, unet, across_timesteps):   
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super().__init__()
        self.across_timesteps = across_timesteps 
        self.unet = unet
        self.get_empty_store()



def make_attention_visualization(org_img: Image, attn_store, track_ids, res=16): 
    # for key, attn in attn_store.step_store.items(): 
        # print(f"{attn.shape = }") 
    # sys.exit(0) 
    print(f"making attention visualization...") 
    total_attn = {} 
    for track_idx in track_ids: 
        total_attn[track_idx] = torch.zeros((res, res)) 
    for key, attns in attn_store.step_store.items(): 
        for attn in attns: 
            if attn.shape[-1] != 77: 
                continue 

            # this is cross attention 
            attn_uncond, attn_cond = attn 

            # this is not the resolution we asked for 
            if attn_cond.shape[-2] != res * res: 
                continue 

            attn_cond = torch.mean(attn_cond, dim=0).reshape((res, res, 77)).permute(2, 0, 1)  

            for track_idx in track_ids: 
                total_attn[track_idx] = total_attn[track_idx] + attn_cond[track_idx]  

 

def show_image_relevance(image_relevance, image: Image, relevance_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevance_res ** 2, relevance_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevance_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevance_res ** 2, relevance_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis