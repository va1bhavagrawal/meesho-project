from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import math 
import numpy as np 
import PIL 
from PIL import Image 

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm, SD35AdaLayerNormZeroX

from utils import * 

NUM_IMAGE_TOKENS = 4096 

# completely batched inputs and batched outputs 
def visualize_text_to_image_attention(attn, prompts_tokens_strs, timestep, layer_name, images, save_dirs, selected_token_indices=None):  
    if attn.shape[0] == len(prompts_tokens_strs) * 2: # there is unconditional text as well  
        attn = attn[len(prompts_tokens_strs) :] 
    attn = torch.mean(attn, dim=1) # average across heads  
    for batch_idx in range(len(prompts_tokens_strs)): 
        if selected_token_indices is None: 
            selected_token_indices_batch = list(range(len(prompts_tokens_strs[batch_idx])))  
        else: 
            selected_token_indices_batch = selected_token_indices[batch_idx] 
        img = images[batch_idx].resize((256, 256)) 
        # prompt_filename = prompts_filenames[batch_idx]
        save_dir = save_dirs[batch_idx] 
        print(f"visualizing for {layer_name}, {selected_token_indices_batch = }") 
        for seq_idx in selected_token_indices_batch: 
            attn_map = attn[batch_idx][NUM_IMAGE_TOKENS + seq_idx][:NUM_IMAGE_TOKENS].reshape(64, 64) 
            assert attn_map.shape == (64, 64) 
            attn_img = show_image_relevance(attn_map, img, 16) 
            # os.makedirs(output_dir, exist_ok=True) 
            # save_dir = osp.join(output_dir, prompt_filename) 
            # os.makedirs(save_dir, exist_ok=True) 
            img_name = f"t2i__{layer_name}__{str(timestep).zfill(3)}__{str(seq_idx).zfill(3)}__{prompts_tokens_strs[batch_idx][seq_idx]}"  
            save_path = osp.join(save_dir, sanitize_filename(img_name) + ".jpg")    
            attn_img.save(save_path)   


def visualize_image_to_text_attention(attn, prompts_tokens_strs, timestep, layer_name, images, save_dirs, selected_token_indices=None):  
    if attn.shape[0] == len(prompts_tokens_strs) * 2: # there is unconditional text as well  
        attn = attn[len(prompts_tokens_strs) :] 
    attn = torch.mean(attn, dim=1) # average across heads  
    # print(f"{attn.shape = }, {layer_name = }")
    for batch_idx in range(len(prompts_tokens_strs)): 
        if selected_token_indices is None: 
            selected_token_indices_batch = list(range(len(prompts_tokens_strs[batch_idx])))  
        else: 
            selected_token_indices_batch = selected_token_indices[batch_idx] 
        img = images[batch_idx].resize((256, 256)) 
        # prompt_filename = prompts_filenames[batch_idx]
        save_dir = save_dirs[batch_idx] 
        print(f"visualizing for {layer_name}, {selected_token_indices_batch = }") 
        for seq_idx in selected_token_indices_batch: 
            # print(f"{attn.shape = }, {layer_name = }")
            attn_map = attn[batch_idx, :NUM_IMAGE_TOKENS, NUM_IMAGE_TOKENS + seq_idx].reshape(64, 64) 
            assert attn_map.shape == (64, 64) 
            attn_img = show_image_relevance(attn_map, img, 16) 
            # save_dir = osp.join(output_dir, prompt_filename) 
            # os.makedirs(save_dir, exist_ok=True) 
            img_name = f"i2t__{layer_name}__{str(timestep).zfill(3)}__{str(seq_idx).zfill(3)}__{prompts_tokens_strs[batch_idx][seq_idx]}"  
            save_path = osp.join(save_dir, sanitize_filename(img_name) + ".jpg")    
            attn_img.save(save_path)   


class OnlineVisualizeJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer_name: str, timestep: int, prompts_tokens_strs: List[str], images: list, save_dirs: list):  
        """
        timestep: because the attention processor is meant for a "use-and-throw" purpose, the timestep can be declared during initialization. 
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.layer_name = layer_name 
        self.timestep = timestep 
        self.prompts_tokens_strs = prompts_tokens_strs 
        self.images = images 
        self.save_dirs = save_dirs 


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # print(f"{query.shape = }") # this is (B * 2, 24, 4429, 64)  
        # print(f"{key.shape = }") # this is the same as query  
        # print(f"{value.shape = }") # this is also the same as query 
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        scale_factor = 1 / math.sqrt(query.size(-1)) 
        attention_weights = (query @ key.transpose(-1, -2)) * scale_factor  
        attention_weights = torch.softmax(attention_weights, dim=-1) 
        hidden_states = (attention_weights @ value).to(query.dtype).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  

        assert len(self.prompts_tokens_strs) == len(self.save_dirs)  
        assert attention_weights.shape[0] == len(self.prompts_tokens_strs) or attention_weights.shape[0] == len(self.prompts_tokens_strs) * 2 
        visualize_text_to_image_attention(attention_weights, self.prompts_tokens_strs, self.timestep, self.layer_name, self.images, self.save_dirs)  
        visualize_image_to_text_attention(attention_weights, self.prompts_tokens_strs, self.timestep, self.layer_name, self.images, self.save_dirs)  
        del attention_weights 
        torch.cuda.empty_cache() 

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states




class VisualizeJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attn_store: dict, layer_name: str):  
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.layer_name = layer_name 
        self.attn_store = attn_store 

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # print(f"{query.shape = }") # this is (B * 2, 24, 4429, 64)  
        # print(f"{key.shape = }") # this is the same as query  
        # print(f"{value.shape = }") # this is also the same as query 
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        scale_factor = 1 / math.sqrt(query.size(-1)) 
        attention_weights = (query @ key.transpose(-1, -2)) * scale_factor  
        attention_weights = torch.softmax(attention_weights, dim=-1) 
        hidden_states = (attention_weights @ value).to(query.dtype).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  
        self.attn_store[self.layer_name] = attention_weights.cpu().detach()   
        del attention_weights 
        torch.cuda.empty_cache() 

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def set_custom_attn_processor(parent_name, parent_module, attn_processor_class=VisualizeJointAttnProcessor2_0, **attn_processor_kwargs):  
    r"""
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            for **all** `Attention` layers.

            If `processor` is a dict, the key needs to define the path to the corresponding cross attention
            processor. This is strongly recommended when setting trainable attention processors.

    """
    attn_store = {} 
    if attn_processor_class == VisualizeJointAttnProcessor2_0: 
        attn_processor_kwargs["attn_store"] = attn_store  

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, attn_processor_class, **attn_processor_kwargs): 
        if hasattr(module, "set_processor"):
            if "Visualize" in attn_processor_class.__name__: 
                attn_processor_kwargs["layer_name"] = name 
            processor = attn_processor_class(**attn_processor_kwargs) 
            module.set_processor(processor)

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, attn_processor_class, **attn_processor_kwargs) 

    # for name, module in parent_module.named_children():
    #     fn_recursive_attn_processor(name, module, attn_processor_class, **attn_processor_kwargs)  

    fn_recursive_attn_processor(parent_name, parent_module, attn_processor_class, **attn_processor_kwargs)  
    return {
        "attn_store": attn_store, 
    }