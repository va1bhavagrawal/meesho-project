import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union, Dict 

import torch
import torch.nn.functional as F
from torch import nn

import diffusers 
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention 


class AttnProcessor2_0_edited:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

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
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

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


def patch_custom_attention(unet): 
    attn_procs = {}
    cross_att_count = 0
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
        attn_procs[name] = AttnProcessor2_0_edited()  

    unet.set_attn_processor(attn_procs) 