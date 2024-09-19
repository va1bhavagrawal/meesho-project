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
    def __init__(self, name, attn_store, loss_store):
        super().__init__()
        self.name = name 
        self.attn_store = attn_store 
        self.loss_store = loss_store 
        assert loss_store is not None 


    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
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

        # print(f"AT LEAST THE CALL METHOD IS CALLED!")  
        if type(encoder_hidden_states) == dict: 
            # assert len(encoder_hidden_states["attn_assignments"]) == len(encoder_hidden_states["encoder_hidden_states"]) 
            if "p2p" in encoder_hidden_states.keys() and encoder_hidden_states["p2p"] == True:   
                B = len(encoder_hidden_states["attn_assignments"]) 
                # assert B == 4  
                # sys.exit(0) 
                for batch_idx in range(B // 2 + 1, B): 
                    for seq_idx in range(len(key[0])): 
                        key[batch_idx][seq_idx] = key[B // 2][seq_idx] 
                    for seq_idx in range(len(query[0])): 
                        query[batch_idx][seq_idx] = query[B // 2][seq_idx] 
                         
            # print(f"AT LEAST THE ENCODER HIDDEN STATES ARE A DICT!") 
            if (self.loss_store is not None) or ("replace_attn" in encoder_hidden_states.keys() and encoder_hidden_states["replace_attn"] == True):  
                # print(f"CAME HERE!") 
                B = len(encoder_hidden_states["attn_assignments"]) 
                for batch_idx in range(B): 
                    for idx1, idx2 in used_attention_maps[batch_idx].items(): 
                        assert idx1 != idx2 
                        if "replace_attn" in encoder_hidden_states.keys() and encoder_hidden_states["replace_attn"] == True: 
                            key[batch_idx][idx1] = key[batch_idx][idx2] 
                        if self.loss_store is not None: 
                            loss = torch.mean((key[batch_idx][idx1] - key[batch_idx][idx2].detach()) ** 2)  
                            self.loss_store(loss) 

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # print(f"{attention_probs.shape = }") 
        # print(f"{encoder_hidden_states.shape = }")
        # print(f"{hidden_states.shape = }")
        # print(f"{query.shape = }")
        # print(f"{key.shape = }")
        # print(f"{attention_probs.shape = }")
        # sys.exit(0) 
        if self.attn_store is not None: 
            self.attn_store(attention_probs, self.name) 

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


def patch_custom_attention(unet, store_attn, across_timesteps, store_loss): 
    attn_procs = {}
    attn_store = None 
    loss_store = None 
    if store_attn: 
        attn_store = AttentionStore(unet, across_timesteps)  
    if store_loss: 
        loss_store = AttentionLossStore() 
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
        attn_procs[name] = AttendExciteAttnProcessor(name, attn_store, loss_store)  

    unet.set_attn_processor(attn_procs) 

    retval = [] 
    if store_attn: 
        retval.append(attn_store) 
    if store_loss: 
        retval.append(loss_store) 
    return retval 


class AttentionLossStore:
    def get_empty_store(self):
        self.step_store = {"loss": 0.0} 


    def forward(self, loss):
        self.step_store["loss"] = self.step_store["loss"] + loss 


    def __call__(self, loss):
        self.forward(loss) 


    def __init__(self):   
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        self.get_empty_store()


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
            # if layer_name in self.step_store.keys(): 
            #     assert self.step_store[layer_name][0].shape == attn.shape 
            #     if not self.across_timesteps: 
            #         self.step_store[layer_name] = [] 
            # else: 
            #     self.step_store[layer_name] = [] 
            #     if not self.across_timesteps: 
            #         self.step_store[layer_name].append(torch.zeros_like(attn).cpu())  

            # if not self.across_timesteps:  
            #     self.step_store[layer_name][0] = self.step_store[layer_name][0] + attn.cpu()  
            # else: 
            #     self.step_store[layer_name].append(attn)  
            if not self.across_timesteps:  
                self.step_store[layer_name] = [attn.detach().cpu()]   
            elif layer_name not in self.step_store.keys():  
                self.step_store[layer_name] = [attn.detach().cpu()] 
            else: 
                self.step_store[layer_name].append(attn.detach().cpu()) 

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


@torch.no_grad   
def get_attention_maps(attn_store, track_ids, uncond_attn_also, res, batch_size): 
    # for key, attn in attn_store.step_store.items(): 
        # print(f"{attn.shape = }") 
    # sys.exit(0) 
    all_batches_attn_maps = [] 
    for batch_idx in range(batch_size): 
        total_attn = {} 
        for track_idx in track_ids: 
            total_attn[track_idx] = {}  
            # for name in attn_store.step_store.keys(): 
            #     total_attn[track_idx][name] = [] 

            n_timesteps = len(list(attn_store.step_store.values())[0])    
            for name, attns in attn_store.step_store.items(): 
                # if attns[0].shape[-2] != res * res: 
                #     # print(f"{attns[0].shape = }, skipping...") 
                #     continue 
                assert len(attns) == n_timesteps, f"{len(attns) = }, {n_timesteps = }" 
                for attn in attns: 
                    if attn.shape[-1] != 77: 
                        continue 

                    # this is cross attention 
                    if uncond_attn_also: 
                        attn_uncond, attn_cond = torch.chunk(attn, 2, dim=0)  
                    else: 
                        attn_cond = attn 

                    # this is not the resolution we asked for 
                    if attn_cond.shape[-2] != res * res: 
                        continue 

                    # print(f"finally, {attn_cond.shape = }") 
                    # attn_cond = torch.mean(attn_cond, dim=0).reshape((res, res, 77)).permute(2, 0, 1)  
                    # print(f"{attn_cond.shape = }") 
                    attn_cond_split = torch.chunk(attn_cond, chunks=batch_size, dim=0) 
                    attn_cond_batchitem = attn_cond_split[batch_idx] 
                    attn_cond_mean = torch.mean(attn_cond_batchitem, dim=0).reshape((res, res, 77)).permute(2, 0, 1) 

                    if not name in total_attn[track_idx].keys(): 
                        total_attn[track_idx][name] = [] 

                    total_attn[track_idx][name].append(attn_cond_mean[track_idx])  


        assert len(total_attn.keys()) == len(track_ids), f"{total_attn.keys() = }, {track_ids = }" 
        total_attn_ = {} 
        for track_idx in track_ids: 
            total_attn_[track_idx] = [] 
            for name in total_attn[track_idx].keys(): 
                assert len(total_attn[track_idx][name]) == n_timesteps, f"{len(total_attn[track_idx][name]) = }, {n_timesteps = }, {name = }, {track_idx = }" 


        final_attn_maps = {} 
        for track_idx in track_ids: 
            final_attn_maps[track_idx] = []  
            for timestep in range(n_timesteps): 
                track_attn_timestep = torch.tensor([0.0]) 
                for name in total_attn[track_idx].keys(): 
                    assert len(total_attn[track_idx][name]) == n_timesteps, f"{len(total_attn[track_idx][name]) = }, {n_timesteps = }" 
                    track_attn_timestep = track_attn_timestep + total_attn[track_idx][name][timestep] 
                track_attn_timestep = track_attn_timestep / len(total_attn[track_idx].keys()) 
                final_attn_maps[track_idx].append(track_attn_timestep) 
            
        # a dict containing a list of attention maps (for each timestep) for each special token  
        all_batches_attn_maps.append(final_attn_maps) 

    return all_batches_attn_maps  

 

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
    return Image.fromarray(vis) 