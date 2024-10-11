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
import random 

DEBUG_ATTN = False  
BOX_RESIZING_FACTOR = 1.2 
INFINITY = 1e9


def get_attention_scores(
    attn: Attention, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if attn.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=attn.scale,
    )
    del baddbmm_input

    if attn.upcast_softmax:
        attention_scores = attention_scores.float()

    # attention_probs = attention_scores.softmax(dim=-1)

    # attention_probs = attention_probs.to(dtype)

    return attention_scores 


class CustomAttentionProcessor:
    def __init__(self, name, attn_store, loss_store):
        super().__init__()
        self.name = name 
        self.attn_store = attn_store 
        self.loss_store = loss_store 


    def find_attention_mean(self, attention_map): 
        assert attention_map.ndim == 3, f"{attention_map.shape = }" 
        assert attention_map.shape[-1] == attention_map.shape[-2] 
        spatial_dim = attention_map.shape[-1] 
        mesh_i, mesh_j = torch.meshgrid(torch.arange(spatial_dim), torch.arange(spatial_dim), indexing="ij") 
        mesh_i, mesh_j = mesh_i.to(attention_map), mesh_j.to(attention_map) 
        mean_i, mean_j = torch.sum(mesh_i * attention_map) / torch.sum(attention_map), torch.sum(mesh_j * attention_map) / torch.sum(attention_map)  

        if DEBUG_ATTN: 
            mean_i_ = 0 
            mean_j_ = 0 
            across_heads_attention_map = torch.mean(attention_map, dim=0) 
            for i in range(attention_map.shape[-2]): 
                for j in range(attention_map.shape[-1]): 
                    mean_i_ = mean_i_ + i * across_heads_attention_map[i, j] 
                    mean_j_ = mean_j_ + j * across_heads_attention_map[i, j]  
            mean_i_ = int(mean_i_ / torch.sum(across_heads_attention_map))  
            mean_j_ = int(mean_j_ / torch.sum(across_heads_attention_map)) 
            assert int(mean_i) == mean_i_, f"{mean_i = }, {mean_i_ = }"  
            assert int(mean_j) == mean_j_, f"{mean_j = }, {mean_j_ = }" 
            print(f"passed mean calculation checks!") 
        
        return mean_i, mean_j 


    # def bbox_attn_loss_func(self, bboxes, attn_maps): 
    #     loss = 0.0 
    #     B = len(bboxes) 
    #     assert len(attn_maps) == B 
    #     for batch_idx in range(B): 
    #         for asset_idx in range(len(bboxes[batch_idx])):  
    #             bbox = (bboxes[batch_idx][asset_idx] * INTERPOLATION_SIZE).to(dtype=torch.int32)  
    #             attn_maps_for_asset = attn_maps[batch_idx][asset_idx] 
    #             # print(f"{attn_maps_for_asset.shape = }") 
    #             interpolated_attn_maps = F.interpolate(attn_maps_for_asset.unsqueeze(0), size=INTERPOLATION_SIZE, mode="bilinear", align_corners=True).squeeze()  
    #             assert interpolated_attn_maps.shape[-1] == INTERPOLATION_SIZE 
    #             assert interpolated_attn_maps.shape[-2] == INTERPOLATION_SIZE 
    #             assert torch.all(bbox >= 0) and torch.all(bbox <= INTERPOLATION_SIZE)   
    #             attn_inside_bbox = interpolated_attn_maps[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]  
    #             # print(f"{attn_inside_bbox.shape = }") 
    #             assert attn_inside_bbox.shape == (interpolated_attn_maps.shape[0], bbox[3] - bbox[1], bbox[2] - bbox[0]) 
    #             loss_asset = torch.sum(attn_inside_bbox) / torch.sum(interpolated_attn_maps)  
    #             assert loss_asset <= 1 
    #             loss = loss + loss_asset  
    #     return loss 


    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        if type(encoder_hidden_states) == dict: 
            actual_encoder_hidden_states = encoder_hidden_states["encoder_hidden_states"] 
        else: 
            actual_encoder_hidden_states = encoder_hidden_states 

        key = attn.to_k(actual_encoder_hidden_states)
        value = attn.to_v(actual_encoder_hidden_states) 

        if type(encoder_hidden_states) == dict: 
            if "p2p" in encoder_hidden_states.keys() and encoder_hidden_states["p2p"] == True:   
                B = len(encoder_hidden_states["attn_assignments"]) 
                for batch_idx in range(B // 2 + 1, B): 
                    for seq_idx in range(len(key[0])): 
                        key[batch_idx][seq_idx] = key[B // 2][seq_idx] 
                    for seq_idx in range(len(query[0])): 
                        query[batch_idx][seq_idx] = query[B // 2][seq_idx] 
                         
            kwargs = encoder_hidden_states.keys() 
            class2special = "class2special" in kwargs and encoder_hidden_states["class2special"] == True 
            class2special_detached = "class2special_detached" in kwargs and encoder_hidden_states["class2special_detached"] == True 
            special2class_detached = "special2class_detached" in kwargs and encoder_hidden_states["special2class_detached"] == True 
            special2class = "special2class" in kwargs and encoder_hidden_states["special2class"] == True 
            class2special_soft = "class2special_soft" in kwargs and encoder_hidden_states["class2special_soft"] == True 
            any_replacement = class2special or special2class_detached or special2class or class2special_detached or class2special_soft  
            
            # first performing any replacement operations, and then the attention maps are calculated! 
            if any_replacement:  
                B = len(encoder_hidden_states["attn_assignments"]) 
                for batch_idx in range(B): 
                    for idx1, idx2 in encoder_hidden_states["attn_assignments"][batch_idx].items(): 
                        assert idx1 != idx2 

                        if class2special: 
                            key[batch_idx][idx1] = key[batch_idx][idx2] 

                        elif class2special_detached: 
                            # if DEBUG_ATTN: 
                                # print(f"using class2special_detached!") 
                            key[batch_idx][idx1] = key[batch_idx][idx2].detach()  
                        
                        elif special2class_detached: 
                            # if DEBUG_ATTN: 
                                # print(f"using special2class_detached!")
                            key[batch_idx][idx2] = key[batch_idx][idx1].detach()  

                        elif special2class: 
                            key[batch_idx][idx2] = key[batch_idx][idx1]  

                        elif class2special_soft: 
                            key[batch_idx][idx1] = key[batch_idx][idx1] + key[batch_idx][idx2] 
                        
                        else: 
                            assert False 

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # not using the attention probs at this stage, as these need to be masked 
        attention_scores = get_attention_scores(attn, query, key, attention_mask)

        if type(encoder_hidden_states) == dict and ("bbox_from_class_mean" in encoder_hidden_states.keys()) and encoder_hidden_states["bbox_from_class_mean"] == True:  
            B = len(encoder_hidden_states["attn_assignments"]) 
            attention_scores_batch_split = list(torch.chunk(attention_scores, chunks=B, dim=0))  
            if "bboxes" in encoder_hidden_states.keys(): 
                bboxes = encoder_hidden_states["bboxes"] 
                resize_box = True 
            else: 
                resize_box = False 
                bboxes = [] 
                for batch_idx in range(B): 
                    bboxes_example = [] 
                    for asset_idx, (idx1, idx2) in enumerate(encoder_hidden_states["attn_assignments"][batch_idx].items()):  
                        assert idx1 != idx2 
                        # format is x1, y1, x2, y2 
                        bboxes_example.append(torch.tensor([0.25, 0.25, 0.75, 0.75]))  
                    bboxes.append(bboxes_example) 

            for batch_idx in range(B): 
                for asset_idx, (idx1, idx2) in enumerate(encoder_hidden_states["attn_assignments"][batch_idx].items()):  
                    assert idx1 != idx2 
                    assert attention_scores_batch_split[batch_idx].ndim == 3  
                    assert attention_scores_batch_split[batch_idx].shape[-1] == 77 
                    attention_scores_idx1 = attention_scores_batch_split[batch_idx][..., idx1] 
                    attention_scores_idx2 = attention_scores_batch_split[batch_idx][..., idx2]  
                    spatial_dim = int(math.sqrt(attention_scores.shape[-2])) 
                    assert spatial_dim * spatial_dim == attention_scores_idx1.shape[-1] == attention_scores_idx2.shape[-1], f"{spatial_dim = }, {attention_scores_idx1.shape = }, {attention_scores_idx1.shape = }" 
                    n_heads = attention_scores_idx1.shape[0] 
                    assert attention_scores_idx2.shape[0] == n_heads 

                    attention_scores_idx1 = attention_scores_idx1.reshape(n_heads, spatial_dim, spatial_dim)  
                    attention_scores_idx2 = attention_scores_idx2.reshape(n_heads, spatial_dim, spatial_dim)  

                    bbox = bboxes[batch_idx][asset_idx] 
                    mean_i, mean_j = torch.round(spatial_dim * ((bbox[1] + bbox[3]) / 2)), torch.round(spatial_dim * ((bbox[0] + bbox[2]) / 2))  
                    h, w = torch.round((bbox[3] - bbox[1]) * spatial_dim), torch.round((bbox[2] - bbox[0]) * spatial_dim) 
                    if resize_box: 
                        h, w = torch.round(h * BOX_RESIZING_FACTOR), torch.round(w * BOX_RESIZING_FACTOR) 
                    h, w = h.to(dtype=torch.long), w.to(dtype=torch.long) 
                    max_side = max(h, w) 
                    h = max_side 
                    w = max_side 
                    random_scaling_factor = 0.75 + random.random() * 0.5  
                    h = h * random_scaling_factor 
                    random_scaling_factor = 0.75 + random.random() * 0.5  
                    w = w * random_scaling_factor 
                    i_min = torch.round(max(torch.tensor(0), mean_i - h // 2)).to(dtype=torch.long) 
                    i_max = torch.round(min(torch.tensor(spatial_dim) - 1, mean_i + h // 2)).to(dtype=torch.long) 
                    j_min = torch.round(max(torch.tensor(0), mean_j - w // 2)).to(dtype=torch.long) 
                    j_max = torch.round(min(torch.tensor(spatial_dim) - 1, mean_j + w // 2)).to(dtype=torch.long)  
                    attention_mask_ = torch.ones_like(attention_scores_idx1).detach()  
                    attention_mask_[:, i_min : i_max, j_min : j_max] = 0 
                    attention_mask_ = attention_mask_ * -INFINITY  

                    attention_scores_idx1 = attention_scores_idx1 + attention_mask_ 
                    attention_scores_idx2 = attention_scores_idx2 + attention_mask_  

                    idx1_mask = torch.zeros((77, ), requires_grad=False).to(device=attention_scores.device)  
                    idx1_mask[idx1] = 1 
                    replacement = attention_scores_batch_split[batch_idx] * (1 - idx1_mask) + attention_scores_idx1.reshape(n_heads, spatial_dim * spatial_dim, 1) * (idx1_mask) 
                    assert replacement.shape == attention_scores_batch_split[batch_idx].shape 
                    attention_scores_batch_split[batch_idx] = replacement  

                    idx2_mask = torch.zeros((77, ), requires_grad=False).to(device=attention_scores.device)  
                    idx2_mask[idx2] = 1  
                    replacement = attention_scores_batch_split[batch_idx] * (1 - idx2_mask) + attention_scores_idx2.reshape(n_heads, spatial_dim * spatial_dim, 1) * (idx2_mask) 
                    assert attention_scores_batch_split[batch_idx].shape == replacement.shape 
                    attention_scores_batch_split[batch_idx] = replacement  

                    assert torch.allclose(attention_scores_batch_split[batch_idx][..., idx1], attention_scores_idx1.flatten(1,)) 
                    assert torch.allclose(attention_scores_batch_split[batch_idx][..., idx2], attention_scores_idx2.flatten(1,))   

            attention_scores = torch.cat(attention_scores_batch_split, dim=0) 
            attention_probs = F.softmax(attention_scores, dim=-1) 

            if self.loss_store is not None: 
                mean_i_attn, mean_j_attn = self.find_attention_mean(attention_scores_idx2)  
                loss = (mean_i_attn - mean_i) ** 2 + (mean_j_attn - mean_j) ** 2 
                loss = loss / (spatial_dim * spatial_dim) 
                self.loss_store(loss) 
        else: 
            attention_probs = F.softmax(attention_scores, dim=-1) 
            # if DEBUG_ATTN: 
            #     dataloader_idx = len(os.listdir("vis_attnmaps")) - 1  
            # for batch_idx in range(B): 
            #     for asset_idx, (idx1, idx2) in enumerate(encoder_hidden_states["attn_assignments"][batch_idx].items()):  
            #         assert idx1 != idx2 
            #         attention_probs_idx1 = attention_probs_batch_split[batch_idx][..., idx1]  
            #         attention_probs_idx2 = attention_probs_batch_split[batch_idx][..., idx2] 
            #         spatial_dim = int(math.sqrt(attention_probs_idx1.shape[-1])) 
            #         assert spatial_dim * spatial_dim == attention_probs_idx1.shape[-1] 
            #         attention_probs_idx1 = attention_probs_idx1.reshape((attention_probs_idx1.shape[0], spatial_dim, spatial_dim)) 
            #         attention_probs_idx2 = attention_probs_idx2.reshape((attention_probs_idx2.shape[0], spatial_dim, spatial_dim)) 

            #         attention_probs_idx1_interp = F.interpolate(attention_probs_idx1.unsqueeze(0), INTERPOLATION_SIZE, mode="bilinear", align_corners=True).squeeze()  
            #         attention_probs_idx2_interp = F.interpolate(attention_probs_idx2.unsqueeze(0), INTERPOLATION_SIZE, mode="bilinear", align_corners=True).squeeze()  

            #         # mean_i, mean_j = self.find_attention_mean(attention_probs_idx2_interp)   
            #         mean_j, mean_i = (bboxes[batch_idx][asset_idx][0] + bboxes[batch_idx][asset_idx][2]) / 2, (bboxes[batch_idx][asset_idx][1] + bboxes[batch_idx][asset_idx][3]) / 2  
            #         mean_j, mean_i = INTERPOLATION_SIZE * mean_j, INTERPOLATION_SIZE * mean_i 
            #         mean_j, mean_i = mean_j.item(), mean_i.item() 

            #         if self.loss_store is not None: 
            #             # apply the centroid forcing loss 
            #             mean_i_attn, mean_j_attn = self.find_attention_mean(attention_probs_idx2_interp) 
            #             loss = (mean_i_attn - mean_i) ** 2 + (mean_j_attn - mean_j) ** 2 
            #             loss = loss / (INTERPOLATION_SIZE * INTERPOLATION_SIZE) 
            #             self.loss_store(loss) 


            #         given_bbox = bboxes[batch_idx][asset_idx] 
            #         h, w = given_bbox[3] - given_bbox[1], given_bbox[2] - given_bbox[0] 
            #         h, w = int(h * INTERPOLATION_SIZE), int(w * INTERPOLATION_SIZE) 
            #         max_side = max(h, w) 
            #         h = max_side 
            #         w = max_side 
            #         # given_bbox_max_side = max(int(INTERPOLATION_SIZE * (given_bbox[2] - given_bbox[0])), int(INTERPOLATION_SIZE * (given_bbox[3] - given_bbox[1]))) 
            #         # assert 0 < given_bbox_max_side < INTERPOLATION_SIZE 
                    
            #         attention_mask_ = torch.zeros((INTERPOLATION_SIZE, INTERPOLATION_SIZE)).to(attention_probs) 
            #         # attention_mask_[mean_i - given_bbox_max_side // 2 : mean_i + given_bbox_max_side // 2, mean_j - given_bbox_max_side // 2 : mean_j + given_bbox_max_side // 2] = 1   
            #         i_min = int(max(0, mean_i - int(BOX_RESIZING_FACTOR * h / 2)))  
            #         i_max = int(min(INTERPOLATION_SIZE - 1, mean_i + int(BOX_RESIZING_FACTOR * h / 2)))  
            #         j_min = int(max(0, mean_j - int(BOX_RESIZING_FACTOR * w / 2)))  
            #         j_max = int(min(INTERPOLATION_SIZE - 1, mean_j + int(BOX_RESIZING_FACTOR * w / 2)))  
            #         attention_mask_[i_min : i_max, j_min : j_max] = 1  

            #         if DEBUG_ATTN: 
            #             viridis = plt.get_cmap('viridis') 
            #             attn_map = torch.mean(attention_probs_idx2_interp, dim=0).detach().cpu().numpy()   
            #             attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) 
            #             img = viridis(attn_map)  
            #             img = (img[:, :, :3] * 255).astype(np.uint8) 
            #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
            #             # img = np.repeat(attn_map[..., None], repeats=3, axis=-1)  
            #             cv2.circle(img, (int(mean_j), int(mean_i)), radius=0, color=(0, 0, 255), thickness=10)  
            #             # cv2.imwrite(osp.join("vis_attnmaps", f"{str(dataloader_idx).zfill(3)}", f"{str(batch_idx).zfill(3)}__{str(asset_idx).zfill(3)}.jpg"), img) 
            #             cv2.rectangle(img, (j_min, i_min), (j_max, i_max), color=(0, 0, 255), thickness=10)  
            #             cv2.imwrite("img.jpg", img) 
            #             time.sleep(1) 

            #             # print(f"{i_max - i_min = }, {h = }, ratio = {(i_max - i_min) / h}") 
            #             # print(f"{j_max - j_min = }, {w = }, ratio = {(j_max - j_min) / w}") 

            #         attention_probs_idx1_interp = attention_probs_idx1_interp * attention_mask_ 
            #         attention_probs_idx2_interp = attention_probs_idx2_interp * attention_mask_ 

            #         attention_probs_idx1_masked = F.interpolate(attention_probs_idx1_interp.unsqueeze(0), attention_probs_idx1.shape[-1], mode="bilinear", align_corners=True).squeeze().reshape(attention_probs_idx1.shape[0], spatial_dim * spatial_dim)  
            #         attention_probs_idx2_masked = F.interpolate(attention_probs_idx2_interp.unsqueeze(0), attention_probs_idx2.shape[-1], mode="bilinear", align_corners=True).squeeze().reshape(attention_probs_idx2.shape[0], spatial_dim * spatial_dim)  
                    
            #         idx1_mask = torch.zeros((77, ), requires_grad=False).to(attention_probs)  
            #         idx1_mask[idx1] = 1  
            #         assert idx1_mask.requires_grad == False 
            #         # TODO weird error with in place replacement, that is coming up when performing in place replacement for idx1, but not for idx2, must check! 
            #         replacement_attn_maps_example = attention_probs_batch_split[batch_idx] * (1 - idx1_mask) + idx1_mask * attention_probs_idx1_masked[..., None].expand(-1, -1, 77)  
            #         # assert replacement_attn_maps_example.requires_grad 
            #         attention_probs_batch_split[batch_idx] = replacement_attn_maps_example  
            #         # attention_probs_batch_split[batch_idx][..., idx1] = attention_probs_idx1_masked 
            #         # assert attention_probs_batch_split[batch_idx].requires_grad == True 
            #         # assert attention_probs_idx2_masked.requires_grad == True 
            #         attention_probs_batch_split[batch_idx][..., idx2] = attention_probs_idx2_masked 
            #         assert torch.allclose(attention_probs_batch_split[batch_idx][..., idx1], attention_probs_idx1_masked)   
            #         assert torch.allclose(attention_probs_batch_split[batch_idx][..., idx2], attention_probs_idx2_masked)   

            # attention_probs = torch.cat(attention_probs_batch_split, dim=0) 

        # the bounding box attention loss 
        # if self.loss_store is not None and type(encoder_hidden_states) == dict:  
        #     attention_probs_batch_split = torch.chunk(attention_probs, chunks=len(encoder_hidden_states["attn_assignments"]), dim=0)  
        #     bboxes = encoder_hidden_states["bboxes"]  
        #     attn_maps = [] 
        #     B = len(encoder_hidden_states["attn_assignments"]) 
        #     for batch_idx in range(B): 
        #         attn_maps_example = [] 
        #         for idx1, idx2 in encoder_hidden_states["attn_assignments"][batch_idx].items(): 
        #             assert idx1 != idx2 
        #             attention_probs_idx1 = attention_probs_batch_split[batch_idx][..., idx1]  
        #             res = int(math.sqrt(attention_probs_idx1.shape[-1])) 
        #             # asserting that it was a perfectly square attention map 
        #             assert attention_probs_idx1.shape[-1] == res * res 
        #             n_heads = attention_probs_idx1.shape[0] 
        #             attention_probs_idx1 = attention_probs_idx1.reshape(n_heads, res, res) 
        #             attn_maps_example.append(attention_probs_idx1) 
        #         attn_maps.append(attn_maps_example) 
        #         assert len(attn_maps_example) == len(bboxes[batch_idx]) 
        #     assert len(attn_maps) == len(bboxes) 
        #     loss = self.bbox_attn_loss_func(bboxes, attn_maps) 
        #     self.loss_store(loss) 

                    
        # print(f"{attention_probs.shape = }") 
        # print(f"{encoder_hidden_states.shape = }")
        # print(f"{hidden_states.shape = }")
        # print(f"{query.shape = }")
        # print(f"{key.shape = }")
        # print(f"{attention_probs.shape = }")
        # sys.exit(0) 

        # if DEBUG_ATTN and type(encoder_hidden_states) == dict:  
        #     attention_probs_batch_split = list(torch.chunk(attention_probs, chunks=B, dim=0))  
        #     for batch_idx in range(B): 
        #         for asset_idx, (idx1, idx2) in enumerate(encoder_hidden_states["attn_assignments"][batch_idx].items()): 
        #             attn_map_idx1 = attention_probs_batch_split[batch_idx][..., idx1]  
        #             attn_map_idx2 = attention_probs_batch_split[batch_idx][..., idx2]  
        #             spatial_dim = int(math.sqrt(attn_map_idx1.shape[-1]))  
        #             n_heads = attn_map_idx1.shape[0] 
        #             attn_map_idx1 = attn_map_idx1.reshape(n_heads, spatial_dim, spatial_dim) 
        #             attn_map_idx2 = attn_map_idx2.reshape(n_heads, spatial_dim, spatial_dim) 
        #             attn_map_idx1_interp = F.interpolate(attn_map_idx1.unsqueeze(0), INTERPOLATION_SIZE, mode="bilinear", align_corners=True).squeeze()  
        #             attn_map_idx2_interp = F.interpolate(attn_map_idx2.unsqueeze(0), INTERPOLATION_SIZE, mode="bilinear", align_corners=True).squeeze()  

        #             viridis = plt.get_cmap('viridis') 
        #             attn_map = torch.mean(attn_map_idx1_interp, dim=0).detach().cpu().numpy()   
        #             attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) 
        #             img = viridis(attn_map)  
        #             img = (img[:, :, :3] * 255).astype(np.uint8) 
        #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
        #             # img = np.repeat(attn_map[..., None], repeats=3, axis=-1)  
        #             # cv2.circle(img, (mean_j, mean_i), radius=0, color=(0, 0, 255), thickness=20)  
        #             # cv2.rectangle(img, (j_min, i_min), (j_max, i_max), color=(0, 0, 255), thickness=10)  
        #             # cv2.imwrite(osp.join("vis_attnmaps", f"{str(dataloader_idx).zfill(3)}", f"{str(batch_idx).zfill(3)}__{str(asset_idx).zfill(3)}.jpg"), img) 
        #             cv2.imwrite("img.jpg", img) 
        #             time.sleep(1)  

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
        # attn_procs[name] = AttendExciteAttnProcessor(name, attn_store, loss_store)  
        attn_procs[name] = CustomAttentionProcessor(name, attn_store, loss_store)  

    unet.set_attn_processor(attn_procs) 

    retval = {} 
    retval["attn_store"] = attn_store 
    retval["loss_store"] = loss_store 
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
        for track_idx in track_ids[batch_idx]: 
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


        assert len(total_attn.keys()) == len(track_ids[batch_idx]), f"{total_attn.keys() = }, {track_ids[batch_idx] = }" 
        total_attn_ = {} 
        for track_idx in track_ids[batch_idx]: 
            total_attn_[track_idx] = [] 
            for name in total_attn[track_idx].keys(): 
                assert len(total_attn[track_idx][name]) == n_timesteps, f"{len(total_attn[track_idx][name]) = }, {n_timesteps = }, {name = }, {track_idx = }" 


        final_attn_maps = {} 
        for track_idx in track_ids[batch_idx]: 
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