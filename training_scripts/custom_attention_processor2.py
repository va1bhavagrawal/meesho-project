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
import matplotlib.pyplot as plt 
import os 
import os.path as osp 
import time 

DEBUG_ATTN = False  
BOX_RESIZING_FACTOR = 1.2 
INFINITY = 1e9
NUM_TEXT_EMBEDDINGS = 77 
LAMBDA = 1.2 

class CALLAttnProcessor: 
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, layer_name):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.layer_name = layer_name 

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        # print(f"{type(text_conditioning) = }")
        text_conditioning = encoder_hidden_states 
        del encoder_hidden_states 
        if text_conditioning is None: 
            # self attention 
            encoder_hidden_states = None 
        else: 
            encoder_hidden_states = text_conditioning["encoder_hidden_states"] 

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

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

        key = attn.to_k(encoder_hidden_states) 
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # print(f"{query.shape = }")
        # print(f"{key.shape = }")
        # print(f"{value.shape = }")
        scale_factor = 1 / math.sqrt(query.size(-1)) 
        attention_scores = (query @ key.transpose(-1, -2)) * scale_factor  
        if text_conditioning is not None:  
            # cross attention 
            spatial_dim = int(math.sqrt(attention_scores.size(-2)))  

            bsz = query.size(0) 
            attention_scores = attention_scores.reshape((bsz, attn.heads, spatial_dim, spatial_dim, NUM_TEXT_EMBEDDINGS)) 

            for batch_idx in range(bsz): 
                for subjects_info in text_conditioning["subjects_info"][batch_idx]: 
                    # getting the special and subject token idxs 
                    special_token_idx = subjects_info["special_token_idx"] 
                    subject_token_idx = subjects_info["subject_token_idx"] 
                    bbox = subjects_info["bbox"] 

                    # we have the tl and br coordinates 
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox 
                    # x1, y1, x2, y2 = int(x1_norm * spatial_dim), int(y1_norm * spatial_dim), int(x2_norm * spatial_dim), int(y2_norm * spatial_dim) 
                    cx, cy = (x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2 
                    h, w = y2_norm - y1_norm, x2_norm - x1_norm 
                    a = max(h, w) 
                    a = a * LAMBDA 
                    x1, y1, x2, y2 = cx - a / 2, cy - a / 2, cx + a / 2, cy + a / 2  
                    x1, y1, x2, y2 = int(x1 * spatial_dim), int(y1 * spatial_dim), int(x2 * spatial_dim), int(y2 * spatial_dim) 
                    x1 = max(0, x1) 
                    y1 = max(0, y1) 
                    x2 = min(spatial_dim - 1, x2) 
                    y2 = min(spatial_dim - 1, y2) 

                    # preparing the attention mask 
                    attention_mask = torch.ones((spatial_dim, spatial_dim)).to(query.device) * -INFINITY  
                    attention_mask[y1:y2, x1:x2] = 0  

                    # masking out the special and subject tokens w the same attention mask 
                    attention_scores[batch_idx, :, :, :, special_token_idx] = attention_scores[batch_idx, :, :, :, special_token_idx] + attention_mask 
                    attention_scores[batch_idx, :, :, :, subject_token_idx] = attention_scores[batch_idx, :, :, :, subject_token_idx] + attention_mask 


            # for batch_idx in range(bsz): 
            #     n_subjects = len(text_conditioning["subjects_info"][batch_idx]) 
            #     fig, axs = plt.subplots(n_subjects, 2, figsize=(10, 10)) 
            #     for subject_idx, subjects_info in enumerate(text_conditioning["subjects_info"][batch_idx]):  
            #         special_token_idx = subjects_info["special_token_idx"] 
            #         subject_token_idx = subjects_info["subject_token_idx"] 
            #         if n_subjects == 4: 
            #             axs_ = axs[subject_idx ]
            #         else: 
            #             axs_ = axs 
            #         attn_special_token = attention_scores[batch_idx, :, :, :, special_token_idx].mean(dim=0).detach().cpu().numpy() 
            #         attn_special_token = (attn_special_token - attn_special_token.min()) / (attn_special_token.max() - attn_special_token.min()) 
            #         attn_subject_token = attention_scores[batch_idx, :, :, :, subject_token_idx].mean(dim=0).detach().cpu().numpy() 
            #         attn_subject_token = (attn_subject_token - attn_subject_token.min()) / (attn_subject_token.max() - attn_subject_token.min()) 
            #         axs_[0].imshow(attn_special_token) 
            #         axs_[0].set_title(special_token_idx) 
            #         axs_[1].imshow(attn_subject_token)  
            #         axs_[1].set_title(subject_token_idx)  
            #     save_dir = osp.join("debug_attn", self.layer_name)  
            #     os.makedirs(save_dir, exist_ok=True) 
            #     proc_id = subjects_info["proc_id"] 
            #     step = subjects_info["step"] 
            #     plt.suptitle(f"process {proc_id} step_idx = {step} batch_idx = {batch_idx}") 
            #     plt.savefig(osp.join(save_dir, f"{str(proc_id).zfill(3)}__{str(step).zfill(3)}__{str(batch_idx).zfill(3)}.jpg"))  
            #     plt.close("all") 


            attention_scores = F.softmax(attention_scores.reshape(bsz, attn.heads, spatial_dim * spatial_dim, NUM_TEXT_EMBEDDINGS), dim=-1) 

        else: 
            attention_scores = F.softmax(attention_scores, dim=-1) 
            
        hidden_states = (attention_scores @ value).to(query.dtype).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  

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


# class CALLAttnProcessor:
#     r"""
#     Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
#     """

#     def __init__(self):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         temb: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
#             deprecate("scale", "1.0.0", deprecation_message)

#         text_conditioning = encoder_hidden_states 
#         del encoder_hidden_states 
#         if text_conditioning is None: 
#             # self attention 
#             encoder_hidden_states = None 
#         else: 
#             # encoder_hidden_states = text_conditioning["encoder_hidden_states"] 
#             encoder_hidden_states = text_conditioning  

#         residual = hidden_states
#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states




def patch_custom_attention(unet): 
    attn_procs = {}
    attn_store = None 
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

        # attn_procs[name] = MyAttentionProcessor()  
        # attn_procs[name] = AttnProcessor2_0_edited(name, attn_store)     
        # if attn_store is None: 
        #     attn_procs[name] = AttnProcessor2_0_edited(name)      
        # else: 
        #     attn_procs[name] = AttendExciteAttnProcessor(name, attn_store) 
        # attn_procs[name] = AttendExciteAttnProcessor(name, attn_store, loss_store)  
        attn_procs[name] = CALLAttnProcessor(name)   

    unet.set_attn_processor(attn_procs) 