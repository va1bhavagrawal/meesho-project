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
    def __init__(self, merged_emb_dim, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, tmp_dir, text_encoder_bypass, bnha_embeds, bs=8):   
        self.merged_emb_dim = merged_emb_dim 
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
        if self.seed is not None: 
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


    def do_it(self, seed, gif_path, prompt, all_subjects_data, include_class_in_prompt=None):    
        with torch.no_grad(): 
            self.seed = seed 
            self.gif_path = gif_path 
            all_encoder_states = [] 
            all_save_paths = [] 

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

                n_samples = len(gif_subject_data[0]["normalized_azimuths"])  

                mlp_embs_video = [] 
                bnha_embs_video = [] 

                for sample_idx in range(n_samples): 
                    mlp_embs_frame = [] 
                    bnha_embs_frame = [] 
                    for subject_data in gif_subject_data:  
                        normalized_azimuth = subject_data["normalized_azimuths"][sample_idx] 
                        sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(self.accelerator.device)  

                        if "pose_type" in subject_data.keys() and subject_data["pose_type"] == "0": 
                            mlp_emb = torch.zeros((1024, )).to(self.accelerator.device) 
                        else: 
                            mlp_emb = self.mlp(sincos.unsqueeze(0)).squeeze()  

                        if "appearance_type" in subject_data.keys() and subject_data["appearance_type"] != "class":  
                            if subject_data["appearance_type"] == "zero": 
                                bnha_emb = torch.zeros((1024, )).to(self.accelerator.device)  
                            elif subject_data["appearance_type"] == "learnt":  
                                bnha_emb = getattr(self.accelerator.unwrap_model(self.bnha_embeds), subject_data["subject"])  
                        else: 
                            bnha_emb = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_data["subject"]]] 

                        mlp_embs_frame.append(mlp_emb) 
                        bnha_embs_frame.append(bnha_emb)  

                    mlp_embs_video.append(torch.stack(mlp_embs_frame, 0)) 
                    bnha_embs_video.append(torch.stack(bnha_embs_frame, 0))  

                mlp_embs_video = torch.stack(mlp_embs_video, 0) 
                bnha_embs_video = torch.stack(bnha_embs_video, 0)  
                merged_embs_video = self.merger(mlp_embs_video, bnha_embs_video) 

                placeholder_text = "a SUBJECT0 " 
                for asset_idx in range(1, len(gif_subject_data)):  
                    placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} " 
                placeholder_text = placeholder_text.strip() 
                assert prompt.find("PLACEHOLDER") != -1 
                template_prompt = prompt.replace("PLACEHOLDER", placeholder_text) 

                assert (include_class_in_prompt is not None) or "include_class_in_prompt" in subject_data.keys()  
                include_class_in_prompt_here = include_class_in_prompt if include_class_in_prompt is not None else subject_data["include_class_in_prompt"]  

                if not include_class_in_prompt_here: 
                    for asset_idx, subject_data in enumerate(gif_subject_data): 
                        assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                        template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]}")   
                else: 
                    for asset_idx, subject_data in enumerate(gif_subject_data):  
                        assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                        template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {subject_data['subject']}") 

                print(f"{template_prompt}") 
                prompt_ids = self.tokenizer(
                    template_prompt, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(self.accelerator.device)  

                for sample_idx in range(n_samples): 
                    for asset_idx, subject_data in enumerate(gif_subject_data): 
                        for token_idx in range(self.merged_emb_dim // 1024): 
                            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = merged_embs_video[sample_idx][asset_idx]  
                    text_embeddings = self.text_encoder(prompt_ids)[0].squeeze() 
                    all_encoder_states.append(text_embeddings) 
                    all_save_paths.append(osp.join(self.tmp_dir, subjects_string, f"{str(sample_idx).zfill(3)}.jpg")) 
                    

            self.accelerator.wait_for_everyone() 
            self.accelerator.print(f"every thread finished generating the encoder hidden states...") 

            if self.accelerator.is_main_process: 
                for save_path in all_save_paths: 
                    os.makedirs(osp.dirname(save_path), exist_ok=True)   
            self.accelerator.wait_for_everyone() 

            dataset = EncoderStatesDataset(all_encoder_states, all_save_paths)  

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
                collect_generated_images(self.tmp_dir, prompt, self.gif_path)  
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
    infer.do_it("output.gif", "a photo of a SUBJECT on a highway", ["sedan", "truck"], 4, "a", "class", True) 