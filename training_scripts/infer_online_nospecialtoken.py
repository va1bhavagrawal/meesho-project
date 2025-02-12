"""
This code is modified from https://github.com/cloneofsimo/lora.
Parts that are highlighted with Adobe License header are protected by the Adobe License
"""

import sys 
import argparse
import hashlib
import itertools
import math
import os
import shutil 
import os.path as osp 
import inspect
from pathlib import Path
from typing import Optional
import pickle 

import copy 

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np 
from io import BytesIO

from utils import * 
from continuous_word_mlp import * 

import matplotlib.pyplot as plt 

sys.path.append(f"..") 
from lora_diffusion import patch_pipe 
# from metrics import MetricEvaluator from safetensors.torch import load_file

# WHICH_MODEL = "penalize_attn__resume0.0001" 
WHICH_MODEL = "replace_attn_maps"  
WHICH_STEP = 300000   
MAX_SUBJECTS_PER_EXAMPLE = 1  
NUM_SAMPLES = 7  

P2P = False  
P2P_CIRCLE = False  
MAX_P2P_TIMESTEP = 50  
ETA = 1.0  

KEYWORD = f"jeep_pickuptruck"     

ACROSS_TIMESTEPS = False  
NUM_INFERENCE_STEPS = 50 

TEXTUAL_INV = None  
TEXTUAL_INV_SUBJECT = "sedan" 
TI_PATH = osp.join("textual_inversion", "textual_inversion_sedan", "vstarsedan.bin") 

from custom_attention_processor import patch_custom_attention, get_attention_maps, show_image_relevance  

TOKEN2ID = {
    "sks": 48136, 
    "bnha": 49336,  
    "pickup truck": 4629, # using the token for "truck" instead  
    "bus": 2840, 
    "cat": 2368, 
    "giraffe": 22826, 
    "horse": 4558,
    "lion": 5567,  
    "elephant": 10299,   
    "jeep": 11286,  
    "motorbike": 33341,  
    "bicycle": 11652, 
    "tractor": 14607,  
    "truck": 4629,  
    "zebra": 22548,  
    "hatchback": 42348, 
    "sedan": 24237, 
    "suv": 15985, 
    "motocross": 34562,  
    "boat": 4440,   
    "ship": 1158,  
    "plane":5363,  
    "helicopter": 11956,   
    "shoe": 7342,  
    "bird": 3329,  
    "sparrow": 22888,  
    "suitcase": 27792,  
    "chair": 4269,  
    "dolphin": 16464, 
    "fish": 2759, 
    "shark": 7980, 
    "man": 786, 
    "camel": 21914, 
    "dog": 1929,  
    "pickup": 15382, 

    # unque tokens 
    "bk": 14083, 
    "ak": 1196, 
    "ck": 868, 
    "dk": 16196, 
    "ek": 2092, 
    "fk": 12410, 
    "gk": 18719, 
} 

# UNIQUE_TOKENS = ["bnha", "sks", "ak", "bk", "ck", "dk", "ek", "fk", "gk"] 
UNIQUE_TOKENS = {  
    "0_0": "bnha", 
    "0_1": "sks", 
    "0_2": "ak", 
    "1_0": "bk", 
    "1_1": "ck", 
    "1_2": "dk", 
    "2_0": "ek", 
    "2_1": "fk", 
    "2_2": "gk", 
} 


from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler, 
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)

import cv2 
from PIL import Image
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms

from pathlib import Path

import random
import re

from continuous_word_mlp import continuous_word_mlp, MergedEmbedding  
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb 



class EncoderStatesDataset(Dataset): 
    def __init__(self, encoder_states, save_paths, attn_assignments, track_ids, interesting_token_strs): 
        assert len(encoder_states) == len(save_paths) == len(track_ids) == len(attn_assignments) == len(interesting_token_strs) > 0  
        self.encoder_states = encoder_states 
        self.save_paths = save_paths 
        self.attn_assignments = attn_assignments 
        self.track_ids = track_ids 
        self.interesting_token_strs = interesting_token_strs 


    def __len__(self): 
       return len(self.encoder_states)  


    def __getitem__(self, index): 
        assert self.save_paths[index] is not None 
        assert self.encoder_states[index] is not None 
        # print(f"dataset is sending {self.encoder_states[index] = }, {self.save_paths[index] = }")
        # (self.encoder_states[index], [self.save_paths[index]]) 
        return (self.encoder_states[index], self.save_paths[index], self.attn_assignments[index], self.track_ids[index], self.interesting_token_strs[index])   


def collate_fn(examples): 
    save_paths = [example[1] for example in examples]  
    encoder_states = torch.stack([example[0] for example in examples], 0)    
    attn_assignments = [example[2] for example in examples] 
    track_ids = [example[3] for example in examples] 
    interesting_token_strs = [example[4] for example in examples] 
    return {
        "save_paths": save_paths, 
        "encoder_states": encoder_states, 
        "attn_assignments": attn_assignments, 
        "track_ids": track_ids, 
        "interesting_token_strs": interesting_token_strs, 
    }


class Infer: 
    def __init__(self, merged_emb_dim, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, tmp_dir, bnha_embeds, store_attn, bs=8):   
        self.merged_emb_dim = merged_emb_dim 
        self.store_attn = store_attn 
        self.accelerator = accelerator 
        self.unet = unet  
        self.bs = bs if ((not self.store_attn) or (not ACROSS_TIMESTEPS)) else 1  
        # self.bs = self.bs if (not P2P) else 2 
        self.text_encoder = text_encoder  
        self.scheduler = scheduler 
        self.vae = vae 
        self.mlp = mlp 
        self.merger = merger 
        self.bnha_embeds = bnha_embeds 
        self.tmp_dir = tmp_dir  

        self.accelerator.wait_for_everyone() 

        if osp.exists(self.tmp_dir) and self.accelerator.is_main_process: 
            shutil.rmtree(f"{self.tmp_dir}") 
        self.accelerator.wait_for_everyone() 
        if store_attn: 
            self.tmp_dir_attn = osp.join(osp.dirname(self.tmp_dir), osp.basename(self.tmp_dir) + "__attn") 
            if osp.exists(self.tmp_dir_attn) and self.accelerator.is_main_process: 
                shutil.rmtree(self.tmp_dir_attn) 
                os.makedirs(self.tmp_dir_attn) 

        self.accelerator.wait_for_everyone() 
        self.tokenizer = tokenizer 

        self.unet = self.accelerator.prepare(self.unet) 
        self.text_encoder = self.accelerator.prepare(self.text_encoder) 
        self.scheduler = self.accelerator.prepare(self.scheduler) 
        self.vae = self.accelerator.prepare(self.vae) 
        self.mlp = self.accelerator.prepare(self.mlp) 
        self.merger = self.accelerator.prepare(self.merger) 
        self.bnha_embeds = self.accelerator.prepare(self.bnha_embeds) 
        # assert not osp.exists(self.gif_name)  


    def generate_images_in_a_batch_and_save_them(self, batch, step_idx): 
        # print(f"{self.accelerator.process_index} is doing {batch_idx = }")
        uncond_tokens = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt", 
        ).input_ids 
        uncond_encoder_states = self.text_encoder(uncond_tokens.to(self.accelerator.device))[0] 
        uncond_assignments = [] 
        for batch_idx in range(batch["encoder_states"].shape[0]): 
            uncond_assignments.append({}) 
        cond_assignments = batch["attn_assignments"] 
        all_assignments = uncond_assignments + cond_assignments 
        # encoder_states, save_paths = batch 
        encoder_states = batch["encoder_states"].to(self.accelerator.device)  
        save_paths = batch["save_paths"]  
        print(f"{self.accelerator.process_index} is doing {save_paths}") 
        # encoder_states = torch.stack(encoder_states).to(self.accelerator.device) 
        B = encoder_states.shape[0] 
        assert encoder_states.shape == (B, 77, 1024) 
        if self.seed is not None: 
            set_seed(self.seed) 
        else: 
            set_seed(get_common_seed() + accelerator.process_index) 
        # latents = torch.randn(1, 4, 64, 64).to(self.accelerator.device, dtype=self.accelerator.unwrap_model(self.vae).dtype).repeat(B, 1, 1, 1)  
        latents = torch.randn(B, 4, 64, 64).to(self.accelerator.device, dtype=self.accelerator.unwrap_model(self.vae).dtype) 
        # with open("best_latents.pt", "rb") as f: 
        #     latents = torch.load(f).repeat(B, 1, 1, 1).to(self.accelerator.device) 
        self.scheduler.set_timesteps(NUM_INFERENCE_STEPS) 
        retval = patch_custom_attention(self.accelerator.unwrap_model(self.unet), store_attn=self.store_attn, across_timesteps=ACROSS_TIMESTEPS, store_loss=False)    
        loss_store = retval["loss_store"] 
        self.attn_store = retval["attn_store"] 
        assert loss_store is None and not ((self.attn_store is not None) ^ self.store_attn)  
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # scaling the latents for the scheduler timestep  
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0) 
            encoder_states_dict = {
                "encoder_hidden_states": concat_encoder_states, 
                "attn_assignments": all_assignments, 
            }
            if self.replace_attn is not None: 
                encoder_states_dict[self.replace_attn] = True 

            assert not (P2P and P2P_CIRCLE), f"both P2P and P2P_CIRLCE can't be true together!" 
            if P2P and t_idx < MAX_P2P_TIMESTEP: 
                encoder_states_dict["p2p"] = True 
            if P2P_CIRCLE and t_idx < MAX_P2P_TIMESTEP: 
                encoder_states_dict["p2p_circle"] = True  
            

            if not self.replace_attn: 
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=concat_encoder_states).sample 
            else: 
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_states_dict).sample 

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, eta=ETA).prev_sample 

        # scale the latents 
        latents = 1 / 0.18215 * latents

        # decode the latents 
        images = self.accelerator.unwrap_model(self.vae).decode(latents.to(self.accelerator.device, dtype=self.accelerator.unwrap_model(self.vae).dtype)).sample 

        # post processing the images and storing them 
        # os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
        save_path_global = osp.join(self.tmp_dir)  
        # print(f"making {self.tmp_dir}") 
        os.makedirs(save_path_global, exist_ok=True) 

        if self.store_attn: 
            # for name, attns in self.attn_store.step_store.items(): 
            #     assert len(attns) == len(self.scheduler.timesteps) 

            attn_maps_batch = get_attention_maps(self.attn_store, batch["track_ids"][batch_idx], uncond_attn_also=True, res=16, batch_size=self.bs)   
            for batch_idx in range(len(batch["save_paths"])):  
                attn_maps = attn_maps_batch[batch_idx] 

                path, tail = osp.split(batch["save_paths"][batch_idx])  
                _, subjects_string = osp.split(path) 
                save_path = osp.join(self.tmp_dir_attn, subjects_string) 
                os.makedirs(save_path, exist_ok=True) 
                pose_idx = int(osp.basename(batch["save_paths"][batch_idx]).replace(f".jpg", ""))  
                # print(f"\n{pose_idx = } for this batch!\n") 

                image = images[batch_idx] 
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image * 255).to(torch.uint8) 
                image = image.cpu().numpy()  
                image = np.transpose(image, (1, 2, 0)) 
                image = np.ascontiguousarray(image) 
                image = Image.fromarray(image) 
                
                assert len(attn_maps.keys()) == len(batch["track_ids"][batch_idx])  

                assert len(batch['track_ids'][batch_idx]) == len(batch['interesting_token_strs'][batch_idx]) 
                # for timestep in range(len(self.scheduler.timesteps)): 
                if ACROSS_TIMESTEPS: 
                    num_timesteps_in_attnstore = len(self.scheduler.timesteps) 
                else: 
                    num_timesteps_in_attnstore = 1  
                for timestep in range(num_timesteps_in_attnstore):  
                    for track_idx_idx, track_idx in enumerate(batch["track_ids"][batch_idx]):  
                        # print(f"{attn_maps[track_idx][timestep].shape = }") 
                        heatmap = show_image_relevance(attn_maps[track_idx][timestep], image, relevance_res=16) 
                        heatmap_captioned = create_image_with_captions([[heatmap]], [[batch["interesting_token_strs"][batch_idx][track_idx_idx]]]) 
                        save_path = osp.join(self.tmp_dir_attn, subjects_string, f"{str(pose_idx).zfill(3)}__{str(timestep).zfill(3)}__{str(track_idx).zfill(3)}.jpg") 
                        heatmap_captioned.save(save_path) 

        # checking on the attention store, which shhould have stored the attention maps for each attention block by now 
        # if self.attn_store is not None: 
            # for batch_idx in range(self.bs): 
                # make_attention_visualization(images[batch_idx], self.attn_store, batch["track_ids"][batch_idx])  

        # for idx, image in zip(ids, images):  
        for idx, image in enumerate(images): 
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image * 255).to(torch.uint8) 
            # generated_images[idx] = image 
            image = image.cpu().numpy()  
            image = np.transpose(image, (1, 2, 0)) 
            image = np.ascontiguousarray(image) 
            # azimuth = idx // n_prompts_per_azimuth 
            # prompt_idx = idx % n_prompts_per_azimuth 
            # prompt = prompts_dataset.prompts[prompt_idx] 

            # add an additional check here to make sure that the subject IS present in the prompt, otherwise there will be a mixup 
            # subject = prompts_dataset.prompt_wise_subjects[prompt_idx]
            # if subject not in prompt:  
            #     # we must insert the subject information in the prompt, so that there is no mixup!
            #     prompt = prompt.replace("bnha", prompts_dataset.prompt_wise_subjects[prompt_idx])    
            # assert prompt.find(subject) != -1 

            # prompt_ = "_".join(prompt.split()) 
            # save_path_prompt = osp.join(save_path_global, prompt_) 
            # os.makedirs(save_path_prompt, exist_ok=True) 
            image = Image.fromarray(image) 
            # image.save(osp.join(save_path_prompt, f"{str(int(azimuth.item())).zfill(3)}.jpg"))  
            image.save(save_paths[idx])  
            # image = Image.fromarray(image) 
            # image.save(osp.join(f"../gpu_imgs/{accelerator.process_index}", f"{str(int(idx.item())).zfill(3)}.jpg")) 


    def do_it(self, seed, gif_path, prompt, all_subjects_data, args):   
        self.replace_attn = args["replace_attn_maps"] 

        normalize_merged_embedding = args["normalize_merged_embedding"] if "normalize_merged_embedding" in args.keys() else False  
        text_encoder_bypass = args["text_encoder_bypass"] if "text_encoder_bypass" in args.keys() else False  
        use_location_conditioning = args["use_location_conditioning"] if "use_location_conditioning" in args.keys() else False  
        include_class_in_prompt = args["include_class_in_prompt"] if "include_class_in_prompt" in args.keys() else False  

        self.accelerator.wait_for_everyone() 
        if osp.exists(self.tmp_dir) and self.accelerator.is_main_process: 
            shutil.rmtree(f"{self.tmp_dir}") 

        self.accelerator.wait_for_everyone() 

        if self.store_attn: 
            self.tmp_dir_attn = osp.join(osp.dirname(self.tmp_dir), osp.basename(self.tmp_dir) + "__attn") 
            if osp.exists(self.tmp_dir_attn) and self.accelerator.is_main_process: 
                shutil.rmtree(self.tmp_dir_attn) 
                os.makedirs(self.tmp_dir_attn) 

        self.accelerator.wait_for_everyone() 

        with torch.no_grad(): 
            self.seed = seed 
            self.gif_path = gif_path 
            if self.store_attn: 
                self.gif_path_attn = self.gif_path 
                self.gif_path_attn = self.gif_path_attn.replace(f"inference_results", f"multistep_attention_visualizations") 
            all_encoder_states = [] 
            all_save_paths = [] 
            all_attn_assignments = [] 
            all_track_ids = [] 
            all_interesting_token_strs = [] 

            for gif_subject_data in all_subjects_data:  
                subjects = [] 
                for subject_data in gif_subject_data: 
                    subjects.append("_".join(subject_data["subject"].split()))  
                subjects_string = "__".join(subjects) 

                unique_strings = []  
                for asset_idx in range(len(gif_subject_data)): 
                    unique_string_subject = "" 
                    for token_idx in range(self.merged_emb_dim // 1024): 
                        unique_string_subject = unique_string_subject + f"{UNIQUE_TOKENS[f'{asset_idx}_{token_idx}']} " 
                    unique_string_subject = unique_string_subject.strip() 
                    unique_strings.append(unique_string_subject) 

                n_samples = len(gif_subject_data[0]["normalized_azimuths"]) - 1   

                mlp_embs_video = [] 
                if self.mlp is not None: 
                    bnha_embs_video = [] 

                for sample_idx in range(n_samples): 
                    mlp_embs_frame = [] 
                    if self.mlp is not None: 
                        bnha_embs_frame = [] 
                    for subject_data in gif_subject_data:  
                        normalized_azimuth = subject_data["normalized_azimuths"][sample_idx] 
                        if use_location_conditioning: 
                            x = subject_data["x"] 
                            y = subject_data["y"] 
                        if self.mlp is not None: 
                            sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(self.accelerator.device)  
                            if "pose_type" in subject_data.keys() and subject_data["pose_type"] == "0": 
                                mlp_emb = torch.zeros((1024, )).to(self.accelerator.device) 
                            else: 
                                mlp_emb = self.mlp(sincos.unsqueeze(0)).squeeze()  
                        else: 
                            # print(f"for {subject_data['subject']}, we are using {normalized_azimuth = }") 
                            pose_input = torch.tensor([normalized_azimuth]).float().to(self.accelerator.device) 
                            if use_location_conditioning: 
                                x_input = torch.tensor([x]).float().to(self.accelerator.device) 
                                y_input = torch.tensor([y]).float().to(self.accelerator.device) 
                                mlp_emb = self.merger(pose_input, x_input, y_input)  
                            else: 
                                mlp_emb = self.merger(pose_input)    
                            assert mlp_emb.shape == (1, self.merged_emb_dim)  
                            mlp_emb = mlp_emb.squeeze() 

                        if self.mlp is not None: 
                            if "appearance_type" in subject_data.keys() and subject_data["appearance_type"] != "class":  
                                if subject_data["appearance_type"] == "zero": 
                                    bnha_emb = torch.zeros((1024, )).to(self.accelerator.device)  
                                elif subject_data["appearance_type"] == "learnt":  
                                    bnha_emb = getattr(self.accelerator.unwrap_model(self.bnha_embeds), subject_data["subject"])  
                            else: 
                                if subject_data["subject"] in TOKEN2ID.keys(): 
                                    bnha_emb = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_data["subject"]]] 
                                else: 
                                    assert subject_data["subject"] == TEXTUAL_INV 
                                    bnha_emb = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[TEXTUAL_INV_SUBJECT]] 
                                    # bnha_emb = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[]

                        mlp_embs_frame.append(mlp_emb) 
                        if self.mlp is not None: 
                            bnha_embs_frame.append(bnha_emb)  

                    mlp_embs_video.append(torch.stack(mlp_embs_frame, 0)) 
                    if self.mlp is not None: 
                        bnha_embs_video.append(torch.stack(bnha_embs_frame, 0))  

                mlp_embs_video = torch.stack(mlp_embs_video, 0) 
                if self.mlp is not None: 
                    bnha_embs_video = torch.stack(bnha_embs_video, 0)  
                    merged_embs_video = self.merger(mlp_embs_video, bnha_embs_video) 
                else: 
                    merged_embs_video = mlp_embs_video 

                placeholder_text = "a SUBJECT0 " 
                for asset_idx in range(1, len(gif_subject_data)):  
                    placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} " 
                placeholder_text = placeholder_text.strip() 
                # assert prompt.find("PLACEHOLDER") != -1 
                template_prompt = prompt.replace("PLACEHOLDER", placeholder_text) 

                assert (include_class_in_prompt is not None) or "include_class_in_prompt" in subject_data.keys()  
                include_class_in_prompt_here = include_class_in_prompt if include_class_in_prompt is not None else subject_data["include_class_in_prompt"]  

                if not include_class_in_prompt_here: 
                    for asset_idx, subject_data in enumerate(gif_subject_data): 
                        # assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                        template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]}")   
                else: 
                    for asset_idx, subject_data in enumerate(gif_subject_data):  
                        # assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                        if subject_data['subject'] in TOKEN2ID.keys(): 
                            pass
                        else: 
                            assert subject_data['subject'] == TEXTUAL_INV 
                            # ti_embedding = torch.load(TI_PATH) 
                            # ti_embedding = ti_embedding[TEXTUAL_INV].squeeze()  
                        template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {subject_data['subject']}") 

                print(f"{template_prompt}") 
                prompt_ids = self.tokenizer(
                    template_prompt, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(self.accelerator.device)  

                track_ids = [] 
                interesting_token_strs = [] 
                for token_pos, token in enumerate(prompt_ids[0]): 
                    if token in TOKEN2ID.values(): 
                        track_ids.append(token_pos)  
                        interesting_token_strs.append(self.tokenizer.decode(token)) 

                for sample_idx in range(n_samples): 
                    for asset_idx, subject_data in enumerate(gif_subject_data): 
                        subject = subject_data["subject"] 
                        for token_idx in range(self.merged_emb_dim // 1024): 
                            replacement_emb = merged_embs_video[sample_idx][asset_idx]  
                            if normalize_merged_embedding: 
                                replacement_emb_norm = torch.linalg.norm(replacement_emb) 
                                org_emb_norm = torch.linalg.norm(self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]]) 
                                replacement_emb = replacement_emb * org_emb_norm / replacement_emb_norm 
                            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = replacement_emb  
                    text_embeddings = self.text_encoder(prompt_ids)[0].squeeze() 
                    all_encoder_states.append(text_embeddings) 
                    all_save_paths.append(osp.join(self.tmp_dir, subjects_string, f"{str(sample_idx).zfill(3)}.jpg")) 
                    # if self.store_attn: 
                    #     all_save_paths_attn.append(osp.join(self.tmp_dir_attn, subjects_string, f"{str(sample_idx).zfill(3)}")) 
                    all_track_ids.append(track_ids) 
                    all_interesting_token_strs.append(interesting_token_strs) 

                    attn_assignments = {} 
                    unique_token_positions = {} 
                    for asset_idx, subject_data in enumerate(gif_subject_data): 
                        for token_idx in range(self.merged_emb_dim // 1024): 
                            unique_token = UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"] 
                            # assert TOKEN2ID[unique_token] in prompt_ids 
                            # print(f"{list(prompt_ids) = }") 
                            # print(f"{TOKEN2ID[unique_token] = }") 
                            # print(f"{TOKEN2ID[unique_token] = }")
                            # print(f"{prompt_ids = }")
                            assert len(prompt_ids) == 1 
                            # unique_token_idx = prompt_ids.squeeze().tolist().index(TOKEN2ID[unique_token]) 
                            # unique_token_positions[f"{asset_idx}_{token_idx}"] = unique_token_idx 
                            # attn_assignments[unique_token_idx] = unique_token_idx + self.merged_emb_dim // 1024 - token_idx  

                            unique_token_idx = prompt_ids.squeeze().tolist().index(TOKEN2ID[subject_data["subject"]])  
                            unique_token_positions[f"{asset_idx}_{token_idx}"] = unique_token_idx 
                            attn_assignments[unique_token_idx] = unique_token_idx 
                    
                    all_attn_assignments.append(attn_assignments) 

                    if text_encoder_bypass: 
                        for unique_token_name, position in unique_token_positions.items(): 
                            text_embeddings[position] = text_embeddings[position] + self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[unique_token_name]]] 
                    

            self.accelerator.wait_for_everyone() 
            self.accelerator.print(f"every thread finished generating the encoder hidden states...") 

            if self.accelerator.is_main_process: 
                for save_path in all_save_paths: 
                    os.makedirs(osp.dirname(save_path), exist_ok=True)   
            self.accelerator.wait_for_everyone() 

            if P2P or P2P_CIRCLE: 
                all_encoder_states_permuted = [] 
                all_save_paths_permuted = [] 
                all_attn_assignments_permuted = [] 
                all_track_ids_permuted = [] 
                all_interesting_token_strs_permuted = [] 

                stride = NUM_SAMPLES - 1  
                for start_idx in range(NUM_SAMPLES - 1): 
                    for sample_idx in range(start_idx, len(all_encoder_states), stride): 
                        all_encoder_states_permuted.append(all_encoder_states[sample_idx]) 
                        all_save_paths_permuted.append(all_save_paths[sample_idx]) 
                        all_attn_assignments_permuted.append(all_attn_assignments[sample_idx]) 
                        all_track_ids_permuted.append(all_track_ids[sample_idx]) 
                        all_interesting_token_strs_permuted.append(all_interesting_token_strs[sample_idx]) 

                all_encoder_states = all_encoder_states_permuted 
                all_save_paths = all_save_paths_permuted 
                all_attn_assignments = all_attn_assignments_permuted 
                all_track_ids = all_track_ids_permuted 
                all_interesting_token_strs = all_interesting_token_strs_permuted 

            dataset = EncoderStatesDataset(all_encoder_states, all_save_paths, all_attn_assignments, all_track_ids, all_interesting_token_strs)     

            dataloader = DataLoader(dataset, batch_size=self.bs, collate_fn=collate_fn)  
            dataloader = self.accelerator.prepare(dataloader)  
            # dataloader = DataLoader(dataset, batch_size=self.bs)  

            self.accelerator.wait_for_everyone() 
            self.accelerator.print(f"every thread finished preparing their dataloaders...") 
            self.accelerator.print(f"starting generation...") 
            for batch_idx, batch in enumerate(dataloader): 
                self.generate_images_in_a_batch_and_save_them(batch, batch_idx)  

            self.accelerator.wait_for_everyone() 
            self.accelerator.print(f"every thread finished their generation, now collecting them to form a gif...") 
            # if not self.store_attn: 
            if self.accelerator.is_main_process: 
                # collect_generated_images(subjects, self.tmp_dir, prompt, "pose+app", self.gif_name)  
                collect_generated_images(self.tmp_dir, prompt, self.gif_path)  
            self.accelerator.wait_for_everyone() 


            if self.store_attn: 
                if self.accelerator.is_main_process: 
                    for subjects_string in os.listdir(self.tmp_dir_attn): 
                        movie = [] 
                        img_names = sorted(os.listdir(osp.join(self.tmp_dir, subjects_string)))  
                        img_paths = [osp.join(self.tmp_dir, subjects_string, img_name) for img_name in img_names] 
                        # generated_imgs = [Image.open(img_path) for img_path in img_paths]  

                        heatmap_names = sorted(os.listdir(osp.join(self.tmp_dir_attn, subjects_string))) 
                        heatmap_paths = [osp.join(self.tmp_dir_attn, subjects_string, heatmap_name) for heatmap_name in heatmap_names] 
                        # heatmaps = [Image.open(heatmap) for heatmap in heatmap_paths]  


                        if ACROSS_TIMESTEPS: 
                            num_timesteps_in_attnstore = len(self.scheduler.timesteps) 
                        else: 
                            num_timesteps_in_attnstore = 1  
                        for timestep in range(num_timesteps_in_attnstore): 
                            # we are building a frame here 
                            all_cols = [] 
                            all_cols_captions = [] 

                            # in each column we would have a specific pose configuration, and the time axis would have the timesteps of generation  
                            for pose_idx in range(n_samples):  
                                # for each pose there would be a column of images, and those images be decided irrespective of the timestep 
                                # the heatmaps corresponding to the different tokens 
                                heatmap_paths_pose_timestep = sorted([heatmap_path for heatmap_path in heatmap_paths if heatmap_path.find(f"{str(pose_idx).zfill(3)}__{str(timestep).zfill(3)}__") != -1])  
                                heatmaps_pose_timestep = [Image.open(heatmap_path) for heatmap_path in heatmap_paths_pose_timestep] 
                                # the generated image by the model 
                                gen_img_path = [img_path for img_path in img_paths if img_path.find(f"{str(pose_idx).zfill(3)}.jpg") != -1] 
                                assert len(gen_img_path) == 1 
                                gen_img = Image.open(gen_img_path[0]) 
                                gen_img = create_image_with_captions([[gen_img]], [["generated image"]]) 
                                gen_img = gen_img.resize(heatmaps_pose_timestep[0].size) 
                                heatmaps_pose_timestep = [img.convert("RGB") for img in heatmaps_pose_timestep] 
                                debug_path = osp.join(f"vis", f"{str(pose_idx).zfill(3)}__{str(timestep).zfill(3)}") 
                                os.makedirs(debug_path, exist_ok=True)   
                                for some_token_idx, heatmap in enumerate(heatmaps_pose_timestep): 
                                    heatmap.save(osp.join(debug_path, f"{some_token_idx}.jpg")) 
                                all_cols.append([gen_img] + heatmaps_pose_timestep) 
                                all_cols_captions.append([f"{timestep = }"] * (len(heatmaps_pose_timestep) + 1)) 

                            frame_img = create_image_with_captions(all_cols, all_cols_captions)  
                            movie.append(frame_img)  

                        movie_save_path = osp.join(osp.dirname(self.gif_path_attn), subjects_string + "___" + osp.basename(self.gif_path_attn))  
                        create_gif(movie, movie_save_path, duration=0.1)  
                self.accelerator.wait_for_everyone() 

            self.accelerator.wait_for_everyone() 

            # if self.store_attn and self.accelerator.is_main_process: 
            #     self.accelerator.print(f"removing {self.tmp_dir_attn}") 
            #     if osp.exists(self.tmp_dir_attn): 
            #         shutil.rmtree(self.tmp_dir_attn) 

            # self.accelerator.wait_for_everyone() 

            # if self.accelerator.is_main_process: 
            #     print(f"removing {self.tmp_dir}") 
            #     if osp.exists(self.tmp_dir): 
            #         shutil.rmtree(self.tmp_dir) 

            # self.accelerator.wait_for_everyone() 


if __name__ == "__main__": 
    with torch.no_grad(): 
        args_path = osp.join(f"../ckpts/multiobject/", f"__{WHICH_MODEL}", f"args.pkl") 
        assert osp.exists(args_path) 
        with open(args_path, "rb") as f: 
            args = pickle.load(f) 
        if WHICH_MODEL.find("replace_attn_maps") != -1: 
            args["replace_attn_maps"] = "class2special" 

        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1") 
        pipeline.scheduler = DDIMScheduler.from_config( 
            pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"  
        )
        pose_mlp = None 
        if "pose_only_embedding" not in args.keys(): 
            pose_mlp = continuous_word_mlp(2, 1024) 
            merger = MergedEmbedding(args['appearance_skip_connection'], 1024, 1024, args['merged_emb_dim'])  
        else: 
            merger = PoseEmbedding(args['merged_emb_dim']) 
            print(f"using the PoseEmbedding for merger!") 

        training_state_path = osp.join(f"../ckpts/multiobject/", f"__{WHICH_MODEL}", f"training_state_{WHICH_STEP}.pth") 
        assert osp.exists(training_state_path), f"{training_state_path = }"  
        # lora_weight_path = training_state_path.replace(f"training_state", "lora_weight") 
        # lora_weight_path = lora_weight_path.replace(f"pth", "safetensors") 
        # assert osp.exists(lora_weight_path), f"{lora_weight_path = }"  
        training_state = torch.load(training_state_path) 

        # patch_pipe( 
        #     pipeline, 
        #     lora_weight_path,  
        #     patch_text=True, 
        #     patch_ti=True, 
        #     patch_unet=True, 
        # ) 

        if args['train_unet']: 
            with torch.no_grad(): 
                _, _ = inject_trainable_lora(pipeline.unet, r=args['lora_rank']) 
            unet_state_dict = pipeline.unet.state_dict() 
            pretrained_unet_state_dict = training_state["unet"]["lora"]
            for name, param in unet_state_dict.items(): 
                if name.find("lora") == -1: 
                    continue 
                assert name in pretrained_unet_state_dict.keys()  
                unet_state_dict[name] = pretrained_unet_state_dict[name] 
            pipeline.unet.load_state_dict(unet_state_dict) 

        if "pose_only_embedding" not in args.keys() or not args['pose_only_embedding']: 
            pose_mlp.load_state_dict(training_state["contword"]["model"]) 

        print(f"{merger.state_dict().keys() = }") 
        print(f"{training_state['merger']['model'].keys() = }") 
        merger.load_state_dict(training_state["merger"]["model"], strict=False)  

        accelerator = Accelerator() 

        all_subjects = [
            "jeep", 
            "bus", 
            "motorbike", 
            "lion", 
            "horse", 
            "elephant", 
        ] 
        init_embeddings = {} 
        for subject in all_subjects: 
            init_embeddings[subject] = torch.zeros((1024, )) 

        # appearance_embeds = AppearanceEmbeddings(init_embeddings) 
        # appearance_embeds.load_state_dict(training_state["appearance"]["model"]) 

        replace_attn = args['replace_attn_maps']  
        # replace_attn = True 

        if TEXTUAL_INV is not None: 
            assert osp.exists(TI_PATH) 
            ti_embedding = torch.load(TI_PATH) 
            ti_embedding = ti_embedding[TEXTUAL_INV] 
            ti_embedding = ti_embedding.squeeze() 
            num_added_tokens = pipeline.tokenizer.add_tokens([TEXTUAL_INV])
            special_token_ids = pipeline.tokenizer.encode(TEXTUAL_INV, add_special_tokens=False) 
            assert len(special_token_ids) == 1 
            pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
            pipeline.text_encoder.get_input_embeddings().weight[special_token_ids[0]] = ti_embedding    
            TOKEN2ID[TEXTUAL_INV] = special_token_ids[0] 

        infer = Infer(args['merged_emb_dim'], accelerator, pipeline.unet, pipeline.scheduler, pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, pose_mlp, merger, "tmp", None, store_attn=True, bs=2)  


        subjects = [
            [
                {
                    "subject": "jeep", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
                    "appearance_type": "class", 
                    "x": 0.3, 
                    "y": 0.6,  
                }, 
                {
                    "subject": "pickup truck", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.7, 
                    "y": 0.7,  
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE],  
            # [
            #     {
            #         "subject": "bicycle", 
            #         "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
            #         "appearance_type": "class", 
            #         "x": 0.4, 
            #         "y": 0.6, 
            #     }, 
            #     {
            #         "subject": "suv", 
            #         "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
            #         "x": 0.8, 
            #         "y": 0.9, 
            #     }
            # ][:MAX_SUBJECTS_PER_EXAMPLE],  
            # [
            #     {
            #         "subject": "sedan", 
            #         "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
            #         "appearance_type": "class", 
            #         "x": 0.3, 
            #         "y": 0.5, 
            #     }, 
            #     {
            #         "subject": "suv", 
            #         "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
            #         "x": 0.7, 
            #         "y": 0.7,   
            #     }
            # ][:MAX_SUBJECTS_PER_EXAMPLE],  
            # [
            #     {
            #         "subject": "horse", 
            #         "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
            #         "appearance_type": "class", 
            #         "x": 0.3, 
            #         "y": 0.5, 
            #     }, 
            #     {
            #         "subject": "suv", 
            #         "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
            #         "x": 0.7, 
            #         "y": 0.7,   
            #     }
            # ][:MAX_SUBJECTS_PER_EXAMPLE],  
            # [
            #     {
            #         "subject": "tractor", 
            #         "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
            #         "appearance_type": "class", 
            #         "x": 0.3, 
            #         "y": 0.5, 
            #     }, 
            #     {
            #         "subject": "suv", 
            #         "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
            #         "x": 0.7, 
            #         "y": 0.7,   
            #     }
            # ][:MAX_SUBJECTS_PER_EXAMPLE],  
        ]
        prompts = [
            "a photo of a sedan in a modern city street surrounded by towering skyscrapers and neon lights",  
            "a photo of a sedan in front of the leaning tower of Pisa in Italy",  
            "a photo of a sedan in the streets of Venice, with the sun setting in the background", 
            # "a photo of PLACEHOLDER in front of a serene waterfall with trees scattered around the region, and stones scattered in the region where the water is flowing",  
            # "a photo of PLACEHOLDER in a lush green forest with tall, green trees, stones are scattered on the ground in the distance, the ground is mushy and wet with small puddles of water",  
            # "a photo of PLACEHOLDER in a field of dandelions, with the sun shining brightly, there are snowy mountain ranges in the distance",   
        ]
        for prompt in prompts: 

            if accelerator.is_main_process: 
                seed = random.randint(0, 170904) 
                with open(f"seed.pkl", "wb") as f: 
                    pickle.dump(seed, f) 
                set_seed(seed) 
                # if osp.exists("best_latents.pt"): 
                #     os.remove("best_latents.pt")  
                # latents = torch.randn(1, 4, 64, 64)  
                # with open(f"best_latents.pt", "wb") as f: 
                #     torch.save(latents, f) 

            accelerator.wait_for_everyone() 
            if not accelerator.is_main_process: 
                with open("seed.pkl", "rb") as f: 
                    seed = pickle.load(f) 
            accelerator.wait_for_everyone() 

            infer.do_it(None, osp.join(f"inference_results", f"__{WHICH_MODEL}_{WHICH_STEP}_{MAX_SUBJECTS_PER_EXAMPLE}_{replace_attn}_{KEYWORD}", f"{'_'.join(prompt.split())}_{seed}.gif"), prompt, subjects, args)  


        # subjects = [
        #     [
        #         {
        #             "subject": "lion", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.3, 
        #             "y": 0.6,  
        #         }, 
        #         {
        #             "subject": "tractor", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.7, 
        #             "y": 0.7,  
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        #     [
        #         {
        #             "subject": "tractor", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.4, 
        #             "y": 0.6, 
        #         }, 
        #         {
        #             "subject": "horse", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.8, 
        #             "y": 0.9, 
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        #     [
        #         {
        #             "subject": "truck", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.3, 
        #             "y": 0.5, 
        #         }, 
        #         {
        #             "subject": "suv", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.7, 
        #             "y": 0.7,   
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        # ]
        # prompts = [
        #     "a photo of PLACEHOLDER in a serene corn field, feature vast stretch of crops and a farmhouse in the distance, it is a sunny afternoon",  
        #     "a photo of PLACEHOLDER in a lush green forest with tall, green trees, stones are scattered on the ground in the distance, the ground is mushy and wet with small puddles of water",  
        #     "a photo of PLACEHOLDER in a field of dandelions, with the sun shining brightly, there are snowy mountain ranges in the distance",   
        # ]

        # for prompt in prompts: 

        #     if accelerator.is_main_process: 
        #         seed = random.randint(0, 170904) 
        #         with open(f"seed.pkl", "wb") as f: 
        #             pickle.dump(seed, f) 
        #         set_seed(seed) 
        #         # if osp.exists("best_latents.pt"): 
        #         #     os.remove("best_latents.pt")  
        #         # latents = torch.randn(1, 4, 64, 64)  
        #         # with open(f"best_latents.pt", "wb") as f: 
        #         #     torch.save(latents, f) 

        #     accelerator.wait_for_everyone() 
        #     if not accelerator.is_main_process: 
        #         with open("seed.pkl", "rb") as f: 
        #             seed = pickle.load(f) 
        #     accelerator.wait_for_everyone() 

        #     infer.do_it(None, osp.join(f"inference_results", f"__{WHICH_MODEL}_{WHICH_STEP}_{MAX_SUBJECTS_PER_EXAMPLE}_{replace_attn}_{KEYWORD}", f"{'_'.join(prompt.split())}_{seed}.gif"), prompt, subjects, args)  


        # subjects = [
        #     [
        #         {
        #             "subject": "sedan", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.3, 
        #             "y": 0.6,  
        #         }, 
        #         {
        #             "subject": "tractor", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.7, 
        #             "y": 0.7,  
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        #     [
        #         {
        #             "subject": "jeep", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.4, 
        #             "y": 0.6, 
        #         }, 
        #         {
        #             "subject": "tractor", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.8, 
        #             "y": 0.9, 
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        #     [
        #         {
        #             "subject": "dog", 
        #             "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),  
        #             "appearance_type": "class", 
        #             "x": 0.3, 
        #             "y": 0.5, 
        #         }, 
        #         {
        #             "subject": "sedan", 
        #             "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
        #             "x": 0.7, 
        #             "y": 0.7,   
        #         }
        #     ][:MAX_SUBJECTS_PER_EXAMPLE],  
        # ]
        # prompts = [
        #     "a photo of PLACEHOLDER in a modern city street surrounded by towering skyscrapers and neon lights",  
        #     "a photo of PLACEHOLDER in front of the leaning tower of Pisa in Italy",  
        #     "a photo of PLACEHOLDER in a field of dandelions, with the sun shining brightly, there are snowy mountain ranges in the distance",   
        # ]
        # for prompt in prompts: 

        #     if accelerator.is_main_process: 
        #         seed = random.randint(0, 170904) 
        #         with open(f"seed.pkl", "wb") as f: 
        #             pickle.dump(seed, f) 
        #         set_seed(seed) 
        #         # if osp.exists("best_latents.pt"): 
        #         #     os.remove("best_latents.pt")  
        #         # latents = torch.randn(1, 4, 64, 64)  
        #         # with open(f"best_latents.pt", "wb") as f: 
        #         #     torch.save(latents, f) 

        #     accelerator.wait_for_everyone() 
        #     if not accelerator.is_main_process: 
        #         with open("seed.pkl", "rb") as f: 
        #             seed = pickle.load(f) 
        #     accelerator.wait_for_everyone() 

        #     infer.do_it(None, osp.join(f"inference_results", f"__{WHICH_MODEL}_{WHICH_STEP}_{MAX_SUBJECTS_PER_EXAMPLE}_{replace_attn}_{KEYWORD}", f"{'_'.join(prompt.split())}_{seed}.gif"), prompt, subjects, args)  