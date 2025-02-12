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
# from metrics import MetricEvaluator 
from safetensors.torch import load_file

# WHICH_MODEL = "__nosubject_zeroinit_notext_moresteps"  
# WHICH_STEP = 110000  
# WHICH_MODEL = "__freezeapp_large"   
# WHICH_STEP = 110000  

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
    "sedan": 24237, 
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
UNIQUE_TOKENS = ["bnha", "sks"]  

# DEBUG = False  
# BS = 4   
# # SAVE_STEPS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000] 
# # VLOG_STEPS = [4, 50, 100, 200, 500, 1000]   
# VLOG_STEPS = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000]
# SAVE_STEPS = copy.deepcopy(VLOG_STEPS) 
# NUM_SAMPLES = 18  
# NUM_COLS = 4  



from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion_utils import (
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
    def __init__(self, encoder_states, save_paths): 
        assert len(encoder_states) == len(save_paths) 
        self.encoder_states = encoder_states 
        self.save_paths = save_paths 
        for save_path in self.save_paths: 
            # print(f"making sure that {save_path} exists!")
            os.makedirs(osp.dirname(save_path), exist_ok=True) 


    def __len__(self): 
       return len(self.encoder_states)  


    def __getitem__(self, index): 
        assert self.save_paths[index] is not None 
        assert self.encoder_states[index] is not None 
        # print(f"dataset is sending {self.encoder_states[index] = }, {self.save_paths[index] = }")
        # (self.encoder_states[index], [self.save_paths[index]]) 
        return (self.encoder_states[index], self.save_paths[index]) 


def collate_fn(examples): 
    save_paths = [example[1] for example in examples]  
    encoder_states = torch.stack([example[0] for example in examples], 0)    
    return {
        "save_paths": save_paths, 
        "encoder_states": encoder_states, 
    }


class Infer: 
    def __init__(self, merged_emb_dim, seed, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, tmp_dir, text_encoder_bypass, bnha_embeds, bs=8):   
        self.merged_emb_dim = merged_emb_dim 
        self.seed = seed  
        self.accelerator = accelerator 
        self.unet = unet 
        self.text_encoder = text_encoder  
        self.scheduler = scheduler 
        self.vae = vae 
        self.mlp = mlp 
        self.merger = merger 
        self.text_encoder_bypass = text_encoder_bypass
        self.bnha_embeds = bnha_embeds 
        self.bs = bs 
        self.tmp_dir = tmp_dir  
        if osp.exists(self.tmp_dir) and self.accelerator.is_main_process: 
            shutil.rmtree(f"{self.tmp_dir}") 
        self.tokenizer = tokenizer 

        self.unet = self.accelerator.prepare(self.unet) 
        self.text_encoder = self.accelerator.prepare(self.text_encoder) 
        self.scheduler = self.accelerator.prepare(self.scheduler) 
        self.vae = self.accelerator.prepare(self.vae) 
        self.mlp = self.accelerator.prepare(self.mlp) 
        self.merger = self.accelerator.prepare(self.merger) 
        self.bnha_embeds = self.accelerator.prepare(self.bnha_embeds) 
        # assert not osp.exists(self.gif_name)  


    def generate_images_in_a_batch_and_save_them(self, batch, batch_idx): 
        # print(f"{self.accelerator.process_index} is doing {batch_idx = }")
        uncond_tokens = self.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt", 
        ).input_ids 
        uncond_encoder_states = self.text_encoder(uncond_tokens.to(self.accelerator.device))[0] 
        # encoder_states, save_paths = batch 
        encoder_states = batch["encoder_states"].to(self.accelerator.device)  
        save_paths = batch["save_paths"]  
        print(f"{self.accelerator.process_index} is doing {save_paths}") 
        # encoder_states = torch.stack(encoder_states).to(self.accelerator.device) 
        B = encoder_states.shape[0] 
        assert encoder_states.shape == (B, 77, 1024) 
        set_seed(self.seed) 
        latents = torch.randn(1, 4, 64, 64).to(self.accelerator.device).repeat(B, 1, 1, 1)  
        self.scheduler.set_timesteps(50)
        for t in self.scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # scaling the latents for the scheduler timestep  
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0) 
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=concat_encoder_states).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale the latents 
        latents = 1 / 0.18215 * latents

        # decode the latents 
        images = self.accelerator.unwrap_model(self.vae).decode(latents).sample 

        # post processing the images and storing them 
        # os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
        save_path_global = osp.join(self.tmp_dir)  
        # print(f"making {self.tmp_dir}") 
        os.makedirs(save_path_global, exist_ok=True) 
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


    def do_it(self, gif_path, prompt, subjects, n_samples, pose_type, appearance_type, include_class_in_prompt=False):    
        with torch.no_grad(): 
            self.gif_path = gif_path 
            encoder_states = torch.zeros((n_samples * len(subjects), 77, 1024)) 
            normalized_azimuths = np.arange(n_samples) / n_samples 
            
            save_paths = [] 
            elevation = 0 
            radius = 2.0 
            for azimuth_idx, azimuth in enumerate(normalized_azimuths):  
                self.accelerator.print(f"{azimuth = }")
                sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(azimuth)), torch.cos(2 * torch.pi * torch.tensor(azimuth))]).to(self.accelerator.device)  
                mlp_embs = self.mlp(sincos.unsqueeze(0).repeat(len(subjects), 1))    
                if pose_type == "0": 
                    mlp_embs = torch.zeros_like(mlp_embs).to(self.accelerator.device) 
                # aer = torch.tensor([azimuth, elevation, radius] * 2).to(self.accelerator.device).float()  
                # assert torch.all(aer[0] <= 1)  
                # assert torch.all(aer[1] <= 1) 
                # assert torch.all(aer[3] <= 1) 
                # assert torch.all(aer[4] <= 1) 
                # assert aer[0] == aer[3] 
                # assert aer[1] == aer[4] 
                # assert aer[2] == aer[5] 
                # aer[0:3] = torch.sin(2 * torch.pi * aer[0:3]) 
                # aer[3:6] = torch.cos(2 * torch.pi * aer[3:6]) 
                # mlp_embs = mlp(aer.unsqueeze(0).repeat(len(subjects), 1)) 
                subject_prompts = [] 
                bnha_embs = [] 
                for subject_idx, subject in enumerate(subjects):  
                    assert f"SUBJECT" in prompt 
                    subject_ = "_".join(subject.split()) 

                    assert "bnha" not in subject 
                    assert "sks" not in subject 

                    # subject_without_bnha = subject.replace("bnha", "").strip()  
                    unique_string = "" 
                    for i in range(self.merged_emb_dim // 1024): 
                        unique_string = unique_string + UNIQUE_TOKENS[i] + " " 
                    unique_string = unique_string.strip() 
                    if not include_class_in_prompt: 
                        subject_prompt = prompt.replace(f"SUBJECT", unique_string)  
                    else: 
                        subject_prompt = prompt.replace(f"SUBJECT", f"{unique_string} {subject}")   
                    # self.accelerator.print(f"{subject_prompt = }")

                    if appearance_type == "learnt": 
                        assert self.bnha_embeds is not None 
                        bnha_embs.append(getattr(self.accelerator.unwrap_model(self.bnha_embeds), subject))  
                    elif appearance_type == "class": 
                        bnha_embs.append(self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]]) 
                    elif appearance_type == "zero": 
                        bnha_embs.append(torch.zeros((1024, )).to(self.accelerator.device)) 
                    else: 
                        print(f"{appearance_type = }") 
                        assert False 

                    assert subject_prompt.find(unique_string) != -1 
                    subject_prompts.append(subject_prompt) 
                    save_paths.append(osp.join(self.tmp_dir, subject_, f"{str(azimuth_idx).zfill(3)}.jpg")) 
                    
                bnha_embs = torch.stack((bnha_embs), 0) 
                merged_embs = self.merger(mlp_embs, bnha_embs) 

                assert len(subject_prompts) == len(merged_embs) 
                # assert len(save_paths) == len(merged_embs) 

                for i in range(merged_embs.shape[0]): 
                    if pose_type == "0" and appearance_type == "zero": 
                        refined_prompt = subject_prompts[i].replace(unique_string, "") 
                    else: 
                        refined_prompt = subject_prompts[i] 
                    self.accelerator.print(f"{refined_prompt = }") 
                    tokens = self.tokenizer(
                        refined_prompt, 
                        padding="max_length", 
                        max_length=self.tokenizer.model_max_length,
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids 

                    # self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_embs[i]  
                    for j in range(self.merged_emb_dim // 1024):  
                        self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[j]]] = merged_embs[i][j * 1024 : (j+1) * 1024]  

                    if pose_type != "0" or appearance_type != "zero": 
                        for j in range(self.merged_emb_dim // 1024): 
                            assert TOKEN2ID[UNIQUE_TOKENS[j]] in tokens 

                    # bnha_idx = list(tokens[0]).index(TOKEN2ID["bnha"])  
                    # assert tokens[0][bnha_idx] == TOKEN2ID["bnha"] 
                    # text_encoder_outputs = self.text_encoder(tokens.to(self.accelerator.device))[0].squeeze()   
                    # if self.text_encoder_bypass: 
                    #     text_encoder_outputs[bnha_idx] = text_encoder_outputs[bnha_idx] + self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] 
                    # unique_positions 
                    # if self.text_encoder_bypass: 
                    #     for i in range(self.merged_emb_dim // 1024): 
                    #         text_encoder_outputs[]

                    text_embeddings = self.text_encoder(tokens.to(self.accelerator.device))[0].squeeze() 
                    if pose_type != "0" or appearance_type != "zero":  
                        unique_token_positions = [] 
                        for j in range(self.merged_emb_dim // 1024): 
                            unique_token_positions.append(list(tokens[0]).index(TOKEN2ID[UNIQUE_TOKENS[j]])) 
                        if self.text_encoder_bypass: 
                            for j, position in enumerate(unique_token_positions):  
                                text_embeddings[position] = text_embeddings[position] + self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[j]]] 

                    encoder_states[azimuth_idx * len(subjects) + i] = text_embeddings 


            # for subject in subjects: 
            #     subject_ = "_".join(subject.split()) 
            #     os.makedirs(osp.join(self.tmp_dir, subject_)) 
            #     for azimuth_idx in range(n_samples): 
            #         save_path = osp.join(self.tmp_dir, subject_, f"{azimuth_idx.zfill(3)}.jpg") 
            #         save_paths.append(save_path) 

            self.accelerator.wait_for_everyone() 
            self.accelerator.print(f"every thread finished generating the encoder hidden states...") 

            dataset = EncoderStatesDataset(encoder_states, save_paths)  

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

            if self.accelerator.is_main_process: 
                # collect_generated_images(subjects, self.tmp_dir, prompt, "pose+app", self.gif_name)  
                collect_generated_images(subjects, self.tmp_dir, prompt, self.gif_path)  
            self.accelerator.wait_for_everyone() 

        if self.accelerator.is_main_process: 
            print(f"removing {self.tmp_dir}") 
            shutil.rmtree(self.tmp_dir) 


if __name__ == "__main__": 
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1") 
    pose_mlp = continuous_word_mlp(2, 1024) 
    merged_emb_dim = 1024 
    merger = MergedEmbedding(False, 1024, 1024, merged_emb_dim) 
    accelerator = Accelerator() 
    infer = Infer(merged_emb_dim, 908, accelerator, pipeline.unet, pipeline.scheduler, pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, pose_mlp, merger, "tmp", False, None, 4) 
    infer.do_it("output.gif", "a photo of a SUBJECT on a highway", ["sedan", "truck"], 4, "0", "zero", True) 