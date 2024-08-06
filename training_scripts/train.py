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

import matplotlib.pyplot as plt 
import textwrap 

# from metrics import MetricEvaluator 


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
}
DEBUG = False  
BS = 4  
# SAVE_STEPS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000] 
# VLOG_STEPS = [4, 50, 100, 200, 500, 1000]   
VLOG_STEPS = [10000, 40000, 60000, 80000, 100000]
# SAVE_STEPS = copy.deepcopy(VLOG_STEPS) 
SAVE_STEPS = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]  
NUM_SAMPLES = 18   
NUM_COLS = 4  

from datasets import DisentangleDataset 


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
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import random
import re

from continuous_word_mlp import continuous_word_mlp, MergedEmbedding  
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb 

from datasets import PromptDataset  

"""
ADOBE CONFIDENTIAL
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe 
and its suppliers and are protected by all applicable intellectual 
property laws, including trade secret and copyright laws. 
Dissemination of this information or reproduction of this material is 
strictly forbidden unless prior written permission is obtained from Adobe.
"""

def create_gif(images, save_path, duration=1):
    """
    Convert a sequence of NumPy array images to a GIF.
    
    Args:
        images (list): A list of NumPy array images.
        fps (int): The frames per second of the GIF (default is 1).
        loop (int): The number of times the animation should loop (0 means loop indefinitely) (default is 0).
    """
    frames = []
    for img in images:
        # Convert NumPy array to PIL Image
        # img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil = img 
        # Append to frames list
        frames.append(img_pil)
    
    # Save frames to a BytesIO object
    # bytes_io = BytesIO()
    # frames[0].save(bytes_io, save_all=True, append_images=frames[1:], duration=1000/fps, loop=loop, 
                #    disposal=2, optimize=True, subrectangles=True)
    frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=int(duration * 1000))
    
    # gif_bytes = bytes_io.getvalue()
    # with open("temp.gif", "wb") as f:
    #     f.write(gif_bytes)
    # return gif_bytes 
    return 


def infer(args, step_number, wandb_log_data, accelerator, unet, scheduler, vae, text_encoder, mlp, merger, bnha_embeds=None):  
    if DEBUG: 
        input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight) 
    common_seed = get_common_seed() 
    set_seed(common_seed)  
    text_encoder = copy.deepcopy(text_encoder) 
    # unet = copy.deepcopy(unet) 
    # mlp = copy.deepcopy(mlp) 
    # merger = copy.deepcopy(merger) 
    # if bnha_embeds is not None: 
    #     bnha_embeds = copy.deepcopy(bnha_embeds) 
    with torch.no_grad(): 
        vae.to(accelerator.device) 
        # the list of videos 
        # each item in the list is the video of a prompt at different viewpoints, or just random generations if use_sks=False  
        accelerator.print(f"performing type 1 inference...") 
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer", 
        ) 

        subjects = [
            "bnha pickup truck",
            "bnha motorbike",  
            "bnha horse", 
            "bnha lion", 
            "bnha cat", 
            "bnha elephant", 
            "bnha bus", 
            "bnha jeep", 
        ] 


        subjects = random.sample(subjects, NUM_COLS) 

        # if not use_sks: 
        #     prompts_dataset = PromptDataset(num_samples=6, subjects=)  
        # else: 
        #     prompts_dataset = PromptDataset(num_samples=18)  
        print(f"subjects used for type1 inference are: {subjects}") 
        prompts_dataset1 = PromptDataset(num_samples=NUM_SAMPLES, subjects=subjects) 
        prompts_dataset = prompts_dataset1
        # assert len(prompts_dataset) == 12  
        # assert len(prompts_dataset.subjects) == 3 

        n_prompts_per_azimuth = len(prompts_dataset.subjects) * len(prompts_dataset.template_prompts) 
        # assert n_prompts_per_azimuth == 6 
        encoder_hidden_states = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 77, 1024)).to(accelerator.device).contiguous()  

        # this is the inference where we use the learnt embeddings 
        accelerator.print(f"collecting the encoder hidden states for type 1 inference...") 
        for azimuth in range(prompts_dataset.num_samples): 
            if azimuth % accelerator.num_processes == accelerator.process_index: 
                normalized_azimuth = azimuth / prompts_dataset.num_samples 
                sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(accelerator.device) 
                # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp(sincos.unsqueeze(0)) 
                mlp_embs = mlp(sincos.unsqueeze(0).repeat(len(prompts_dataset.prompts), 1))   
                # if args.textual_inv: 
                if True:  
                    bnha_embs = [] 
                    for i in range(len(prompts_dataset.prompt_wise_subjects)):   
                        subject = prompts_dataset.prompt_wise_subjects[i]  
                        # if "bnha" not in subject: 
                        #     # if this is not a bnha subject, then it is not seen during training, and just put the class embedding for the appearance  
                        #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   
                        #     continue 
                        assert "bnha" in subject 
                        subject_without_bnha = subject.replace("bnha", "").strip()  
                        assert subject_without_bnha in args.subjects 

                        # subject = subject.replace("bnha", "").strip() 

                        # if hasattr(bnha_embs, subject): 
                        # assert hasattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha)  
                            # if the subject (after removing bnha) is in the training subjects, then just replace the learnt appearance embedding 
                        # bnha_embs.append(getattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha))     
                        bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_without_bnha]]) 
                            # bnha_embs.append(bnha_embeds(subject))      
                        # else: 
                        #     # if the subject is not in the training subjects, then zero is passed as the appearance embedding 
                        #     # bnha_embs.append(torch.zeros(1024)) 
                        #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   

                    bnha_embs = torch.stack(bnha_embs)  
                else: 
                    # bnha_embs = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]].detach().unsqueeze(0).repeat(len(prompts_dataset.prompts), 1)  
                    raise NotImplementedError("not implemented the inference case without textual inversion...")  

                merged_embs = merger(mlp_embs, bnha_embs)  

                for i, merged_emb in enumerate(merged_embs):  
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_embs[i]  

                # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = mlp(sincos.unsqueeze(0)) 
                    tokens = tokenizer(
                        prompts_dataset.prompts[i], 
                        padding="max_length", 
                        max_length=tokenizer.model_max_length,
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids 
                    text_encoder_outputs = text_encoder(tokens.to(accelerator.device))[0].squeeze()   
                    encoder_hidden_states[azimuth * n_prompts_per_azimuth + i] = text_encoder_outputs  
        encoder_hidden_states = torch.sum(accelerator.gather(encoder_hidden_states.unsqueeze(0)), dim=0)  

        encoder_states_dataset = torch.utils.data.TensorDataset(encoder_hidden_states, torch.arange(encoder_hidden_states.shape[0]))  

        generated_images = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 3, 512, 512)).to(accelerator.device)  
        encoder_states_dataloader = torch.utils.data.DataLoader(
            encoder_states_dataset, 
            batch_size=args.inference_batch_size,  
            shuffle=False, 
        ) 

        encoder_states_dataloader = accelerator.prepare(encoder_states_dataloader) 

        uncond_tokens = tokenizer(
            [""], 
            padding="max_length", 
            max_length=tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt", 
        ).input_ids 
        uncond_encoder_states = text_encoder(uncond_tokens.to(accelerator.device))[0] 

        torch.manual_seed(args.seed * accelerator.process_index) 
        accelerator.print(f"starting generation for type 1 inference...")  
        for batch in tqdm(encoder_states_dataloader, disable = not accelerator.is_main_process):  
            encoder_states, ids = batch 
            B = encoder_states.shape[0] 
            assert encoder_states.shape == (B, 77, 1024) 
            latents = torch.randn(B, 4, 64, 64).to(accelerator.device)  
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # scaling the latents for the scheduler timestep  
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0) 
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=concat_encoder_states).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # scale the latents 
            latents = 1 / 0.18215 * latents

            # decode the latents 
            images = vae.decode(latents).sample 

            # post processing the images and storing them 
            # os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
            save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", f"type1")  
            os.makedirs(save_path_global, exist_ok=True) 
            for idx, image in zip(ids, images):  
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image * 255).to(torch.uint8) 
                generated_images[idx] = image 
                image = image.cpu().numpy()  
                image = np.transpose(image, (1, 2, 0)) 
                image = np.ascontiguousarray(image) 
                azimuth = idx // n_prompts_per_azimuth 
                prompt_idx = idx % n_prompts_per_azimuth 
                prompt = prompts_dataset.prompts[prompt_idx] 

                # add an additional check here to make sure that the subject IS present in the prompt, otherwise there will be a mixup 
                subject = prompts_dataset.prompt_wise_subjects[prompt_idx]
                if subject not in prompt:  
                    # we must insert the subject information in the prompt, so that there is no mixup!
                    prompt = prompt.replace("bnha", prompts_dataset.prompt_wise_subjects[prompt_idx])    
                assert prompt.find(subject) != -1 

                prompt_ = "_".join(prompt.split()) 
                save_path_prompt = osp.join(save_path_global, prompt_) 
                os.makedirs(save_path_prompt, exist_ok=True) 
                image = Image.fromarray(image) 
                image.save(osp.join(save_path_prompt, f"{str(int(azimuth.item())).zfill(3)}.jpg"))  
                # image = Image.fromarray(image) 
                # image.save(osp.join(f"../gpu_imgs/{accelerator.process_index}", f"{str(int(idx.item())).zfill(3)}.jpg")) 

        # vae = vae.to(torch.device("cpu")) 
        # accelerator.wait_for_everyone() 


        ###################### TYPE 2 INFERENCE ################################################
        ########################################################################################
        # subjects = [
        #     "bnha pickup truck",
        #     "bnha motorbike",  
        #     "bnha horse", 
        #     "bnha lion", 
        # ] 
        # subjects = [
        #     "bicycle", 
        #     "tractor", 
        #     "sports car", 
        #     "brad pitt", 
        # ]

        # IT MAKES SENSE TO KEEP THE SAME SUBJECTS AS THE TYPE 1 INFERENCE 
        # subjects = [
        #     "bnha pickup truck",
        #     "bnha motorbike",  
        #     "bnha horse", 
        #     "bnha lion", 
        #     "bnha cat", 
        #     "bnha elephant", 
        #     "bnha bus", 
        #     "bnha giraffe", 
        #     "bnha jeep", 
        # ] 

        # subjects = random.sample(subjects, NUM_COLS) 

        # if not use_sks: 
        #     prompts_dataset = PromptDataset(num_samples=6, subjects=)  
        # else: 
        #     prompts_dataset = PromptDataset(num_samples=18)  

        # torch.cuda.empty_cache() 

        # common_seed = get_common_seed() 
        # set_seed(common_seed)  

        # prompts_dataset2 = PromptDataset(num_samples=NUM_SAMPLES, subjects=subjects) 
        # prompts_dataset = prompts_dataset2 
        # # assert len(prompts_dataset) == 12  

        # n_prompts_per_azimuth = len(prompts_dataset.subjects) * len(prompts_dataset.template_prompts) 
        # encoder_hidden_states = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 77, 1024)).to(accelerator.device).contiguous()  

        # # this is the inference where we use the learnt embeddings 
        # accelerator.print(f"collecting the encoder hidden states for type 2 inference...") 
        # for azimuth in range(prompts_dataset.num_samples): 
        #     if azimuth % accelerator.num_processes == accelerator.process_index: 
        #         normalized_azimuth = azimuth / prompts_dataset.num_samples 
        #         sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(accelerator.device) 
        #         # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp(sincos.unsqueeze(0)) 
        #         mlp_embs = mlp(sincos.unsqueeze(0).repeat(len(prompts_dataset.prompts), 1))   
        #         if args.textual_inv: 
        #             bnha_embs = [] 
        #             for i in range(len(prompts_dataset.prompt_wise_subjects)):   
        #                 subject = prompts_dataset.prompt_wise_subjects[i]  
        #                 # if "bnha" not in subject: 
        #                 #     # if this is not a bnha subject, then it is not seen during training, and just put the class embedding for the appearance  
        #                 #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   
        #                 #     continue 
        #                 assert "bnha" in subject 
        #                 subject_without_bnha = subject.replace("bnha", "").strip()  
        #                 # this assertion is necessary as type 2 inference has only subjects seen during training 
        #                 assert subject_without_bnha in args.subjects 

        #                 # subject = subject.replace("bnha", "").strip() 

        #                 # if hasattr(bnha_embs, subject): 
        #                 # this assertion is necessary as type 2 inference has only subjects seen during training 
        #                 assert hasattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha)  
        #                     # if the subject (after removing bnha) is in the training subjects, then just replace the learnt appearance embedding 
        #                 # bnha_embs.append(getattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha))     
        #                 # we use the class embeddings instead of learnt embedding in the type2 inference... 
        #                 bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_without_bnha]]) 

        #                 if DEBUG: 
        #                     # to check that the token embedding for the subject did not change, and is same as that for original CLIPTextEncoder 
        #                     assert torch.allclose(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_without_bnha]], input_embeddings[TOKEN2ID[subject_without_bnha]])  

        #                     # to check that the appearance embedding did receive some update!
        #                     assert not torch.allclose(getattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha), accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_without_bnha]]) 

        #                     # to check that the text encoder's bnha embedding and the class embedding are the same 
        #                     assert torch.allclose(bnha_embs[-1], input_embeddings[TOKEN2ID[subject_without_bnha]]) 


        #                     # bnha_embs.append(bnha_embeds(subject))      
        #                 # else: 
        #                 #     # if the subject is not in the training subjects, then zero is passed as the appearance embedding 
        #                 #     # bnha_embs.append(torch.zeros(1024)) 
        #                 #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   

        #             bnha_embs = torch.stack(bnha_embs)  
        #         else: 
        #             # bnha_embs = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]].detach().unsqueeze(0).repeat(len(prompts_dataset.prompts), 1)  
        #             raise NotImplementedError("not implemented the inference case without textual inversion...")  

        #         merged_embs = merger(mlp_embs, bnha_embs)  

        #         for i, merged_emb in enumerate(merged_embs):  
        #             accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_embs[i]  

        #         # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = mlp(sincos.unsqueeze(0)) 
        #             tokens = tokenizer(
        #                 prompts_dataset.prompts[i], 
        #                 padding="max_length", 
        #                 max_length=tokenizer.model_max_length,
        #                 truncation=True, 
        #                 return_tensors="pt"
        #             ).input_ids 
        #             text_encoder_outputs = text_encoder(tokens.to(accelerator.device))[0].squeeze()   
        #             encoder_hidden_states[azimuth * n_prompts_per_azimuth + i] = text_encoder_outputs  
        # encoder_hidden_states = torch.sum(accelerator.gather(encoder_hidden_states.unsqueeze(0)), dim=0)  

        # encoder_states_dataset = torch.utils.data.TensorDataset(encoder_hidden_states, torch.arange(encoder_hidden_states.shape[0]))  

        # generated_images = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 3, 512, 512)).to(accelerator.device)  
        # encoder_states_dataloader = torch.utils.data.DataLoader(
        #     encoder_states_dataset, 
        #     batch_size=args.inference_batch_size,  
        #     shuffle=False, 
        # ) 

        # encoder_states_dataloader = accelerator.prepare(encoder_states_dataloader) 

        # uncond_tokens = tokenizer(
        #     [""], 
        #     padding="max_length", 
        #     max_length=tokenizer.model_max_length,
        #     truncation=True, 
        #     return_tensors="pt", 
        # ).input_ids 
        # uncond_encoder_states = text_encoder(uncond_tokens.to(accelerator.device))[0] 

        # torch.manual_seed(args.seed * accelerator.process_index) 
        # accelerator.print(f"starting generation for type 2 inference...")  
        # for batch in tqdm(encoder_states_dataloader, disable = not accelerator.is_main_process):  
        #     encoder_states, ids = batch 
        #     B = encoder_states.shape[0] 
        #     assert encoder_states.shape == (B, 77, 1024) 
        #     latents = torch.randn(B, 4, 64, 64).to(accelerator.device)  
        #     scheduler.set_timesteps(50)
        #     for t in scheduler.timesteps:
        #         # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #         latent_model_input = torch.cat([latents] * 2)

        #         # scaling the latents for the scheduler timestep  
        #         latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        #         # predict the noise residual
        #         concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0) 
        #         noise_pred = unet(latent_model_input, t, encoder_hidden_states=concat_encoder_states).sample

        #         # perform guidance
        #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        #         # compute the previous noisy sample x_t -> x_t-1
        #         latents = scheduler.step(noise_pred, t, latents).prev_sample

        #     # scale the latents 
        #     latents = 1 / 0.18215 * latents

        #     # decode the latents 
        #     images = vae.decode(latents).sample 

        #     # post processing the images and storing them 
        #     # os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
        #     save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", f"type2")   
        #     os.makedirs(save_path_global, exist_ok=True) 
        #     for idx, image in zip(ids, images):  
        #         image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        #         image = (image * 255).to(torch.uint8) 
        #         generated_images[idx] = image 
        #         image = image.cpu().numpy()  
        #         image = np.transpose(image, (1, 2, 0)) 
        #         image = np.ascontiguousarray(image) 
        #         azimuth = idx // n_prompts_per_azimuth 
        #         prompt_idx = idx % n_prompts_per_azimuth 
        #         prompt = prompts_dataset.prompts[prompt_idx] 

        #         # add an additional check here to make sure that the subject IS present in the prompt, otherwise there will be a mixup 
        #         subject = prompts_dataset.prompt_wise_subjects[prompt_idx]
        #         if subject not in prompt:  
        #             # we must insert the subject information in the prompt, so that there is no mixup!
        #             prompt = prompt.replace("bnha", prompts_dataset.prompt_wise_subjects[prompt_idx])    
        #         assert prompt.find(subject) != -1 

        #         prompt_ = "_".join(prompt.split()) 
        #         save_path_prompt = osp.join(save_path_global, prompt_) 
        #         os.makedirs(save_path_prompt, exist_ok=True) 
        #         image = Image.fromarray(image) 
        #         image.save(osp.join(save_path_prompt, f"{str(int(azimuth.item())).zfill(3)}.jpg"))  
        #         # image = Image.fromarray(image) 
        #         # image.save(osp.join(f"../gpu_imgs/{accelerator.process_index}", f"{str(int(idx.item())).zfill(3)}.jpg")) 

        # # vae = vae.to(torch.device("cpu")) 
        # # accelerator.wait_for_everyone() 


        ###################### TYPE 3 INFERENCE ################################################
        ########################################################################################
        # subjects = [
        #     "bnha pickup truck",
        #     "bnha motorbike",  
        #     "bnha horse", 
        #     "bnha lion", 
        # ] 

        torch.cuda.empty_cache() 


        common_seed = get_common_seed() 
        set_seed(common_seed)  

        subjects = [
            "bnha bicycle", 
            "bnha tractor", 
            "bnha truck", 
            "bnha zebra",  
            "bnha sedan", 
        ]

        subjects = random.sample(subjects, NUM_COLS)  

        # if not use_sks: 
        #     prompts_dataset = PromptDataset(num_samples=6, subjects=)  
        # else: 
        #     prompts_dataset = PromptDataset(num_samples=18)  
        prompts_dataset3 = PromptDataset(num_samples=NUM_SAMPLES, subjects=subjects) 
        prompts_dataset = prompts_dataset3 
        # assert len(prompts_dataset) == 12  

        n_prompts_per_azimuth = len(prompts_dataset.subjects) * len(prompts_dataset.template_prompts) 
        encoder_hidden_states = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 77, 1024)).to(accelerator.device).contiguous()  

        # this is the inference where we use the learnt embeddings 
        accelerator.print(f"collecting the encoder hidden states for type 3 inference...") 
        for azimuth in range(prompts_dataset.num_samples): 
            if azimuth % accelerator.num_processes == accelerator.process_index: 
                normalized_azimuth = azimuth / prompts_dataset.num_samples 
                sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(accelerator.device) 
                # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp(sincos.unsqueeze(0)) 
                mlp_embs = mlp(sincos.unsqueeze(0).repeat(len(prompts_dataset.prompts), 1))   
                # if args.textual_inv: 
                if True:  
                    bnha_embs = [] 
                    for i in range(len(prompts_dataset.prompt_wise_subjects)):   
                        subject = prompts_dataset.prompt_wise_subjects[i]  
                        # if "bnha" not in subject: 
                        #     # if this is not a bnha subject, then it is not seen during training, and just put the class embedding for the appearance  
                        #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   
                        #     continue 
                        # again this assertion is necessary because we want to test pose control here also 
                        assert "bnha" in subject 
                        subject_without_bnha = subject.replace("bnha", "").strip()  
                        # this assertion is necessary as type 3 inference has only subjects NOT seen during training 
                        assert subject_without_bnha not in args.subjects 

                        # subject = subject.replace("bnha", "").strip() 

                        # if hasattr(bnha_embs, subject): 
                        # this assertion is necessary as type 3 inference has only subjects NOT seen during training 
                        # assert not hasattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha)  
                            # if the subject (after removing bnha) is in the training subjects, then just replace the learnt appearance embedding 
                        # bnha_embs.append(getattr(accelerator.unwrap_model(bnha_embeds), subject_without_bnha))     
                        bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject_without_bnha]]) 
                            # bnha_embs.append(bnha_embeds(subject))      
                        # else: 
                        #     # if the subject is not in the training subjects, then zero is passed as the appearance embedding 
                        #     # bnha_embs.append(torch.zeros(1024)) 
                        #     bnha_embs.append(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])   

                    bnha_embs = torch.stack(bnha_embs)  
                else: 
                    # bnha_embs = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]].detach().unsqueeze(0).repeat(len(prompts_dataset.prompts), 1)  
                    raise NotImplementedError("not implemented the inference case without textual inversion...")  

                merged_embs = merger(mlp_embs, bnha_embs)  

                for i, merged_emb in enumerate(merged_embs):  
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_embs[i]  

                # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = mlp(sincos.unsqueeze(0)) 
                    tokens = tokenizer(
                        prompts_dataset.prompts[i], 
                        padding="max_length", 
                        max_length=tokenizer.model_max_length,
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids 
                    text_encoder_outputs = text_encoder(tokens.to(accelerator.device))[0].squeeze()   
                    encoder_hidden_states[azimuth * n_prompts_per_azimuth + i] = text_encoder_outputs  
        encoder_hidden_states = torch.sum(accelerator.gather(encoder_hidden_states.unsqueeze(0)), dim=0)  

        encoder_states_dataset = torch.utils.data.TensorDataset(encoder_hidden_states, torch.arange(encoder_hidden_states.shape[0]))  

        generated_images = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 3, 512, 512)).to(accelerator.device)  
        encoder_states_dataloader = torch.utils.data.DataLoader(
            encoder_states_dataset, 
            batch_size=args.inference_batch_size,  
            shuffle=False, 
        ) 

        encoder_states_dataloader = accelerator.prepare(encoder_states_dataloader) 

        uncond_tokens = tokenizer(
            [""], 
            padding="max_length", 
            max_length=tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt", 
        ).input_ids 
        uncond_encoder_states = text_encoder(uncond_tokens.to(accelerator.device))[0] 

        torch.manual_seed(args.seed * accelerator.process_index) 
        accelerator.print(f"starting generation for type 3 inference...")  
        for batch in tqdm(encoder_states_dataloader, disable = not accelerator.is_main_process):  
            encoder_states, ids = batch 
            B = encoder_states.shape[0] 
            assert encoder_states.shape == (B, 77, 1024) 
            latents = torch.randn(B, 4, 64, 64).to(accelerator.device)  
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # scaling the latents for the scheduler timestep  
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0) 
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=concat_encoder_states).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # scale the latents 
            latents = 1 / 0.18215 * latents

            # decode the latents 
            images = vae.decode(latents).sample 

            # post processing the images and storing them 
            # os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
            save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", f"type3")   
            os.makedirs(save_path_global, exist_ok=True) 
            for idx, image in zip(ids, images):  
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image * 255).to(torch.uint8) 
                generated_images[idx] = image 
                image = image.cpu().numpy()  
                image = np.transpose(image, (1, 2, 0)) 
                image = np.ascontiguousarray(image) 
                azimuth = idx // n_prompts_per_azimuth 
                prompt_idx = idx % n_prompts_per_azimuth 
                prompt = prompts_dataset.prompts[prompt_idx] 

                # add an additional check here to make sure that the subject IS present in the prompt, otherwise there will be a mixup 
                subject = prompts_dataset.prompt_wise_subjects[prompt_idx]
                if subject not in prompt:  
                    # we must insert the subject information in the prompt, so that there is no mixup!
                    prompt = prompt.replace("bnha", prompts_dataset.prompt_wise_subjects[prompt_idx])    
                assert prompt.find(subject) != -1 

                prompt_ = "_".join(prompt.split()) 
                save_path_prompt = osp.join(save_path_global, prompt_) 
                os.makedirs(save_path_prompt, exist_ok=True) 
                image = Image.fromarray(image) 
                image.save(osp.join(save_path_prompt, f"{str(int(azimuth.item())).zfill(3)}.jpg"))  
                # image = Image.fromarray(image) 
                # image.save(osp.join(f"../gpu_imgs/{accelerator.process_index}", f"{str(int(idx.item())).zfill(3)}.jpg")) 

        vae = vae.to(torch.device("cpu")) 
        accelerator.wait_for_everyone() 

        # videos = {} 
        # for prompt_ in os.listdir(save_path_global): 
        #     prompt = " ".join(prompt_.split("_")) 
        #     save_path_prompt = osp.join(save_path_global, prompt_) 
        #     videos[prompt_] = [] 
        #     img_names = os.listdir(save_path_prompt) 
        #     img_names = [img_name for img_name in img_names if img_name.find(f"jpg") != -1] 
        #     img_names = sorted(img_names) 
        #     for img_name in img_names: 
        #         img_path = osp.join(save_path_prompt, img_name) 
        #         img = Image.open(img_path) 
        #         videos[prompt_].append(img) 
        #     video_path = osp.join(save_path_prompt, prompt_ + ".gif") 
        #     create_gif(videos[prompt_], video_path, 1)  
        #     if accelerator.is_main_process and args.wandb:  
        #         wandb_log_data[prompt] = wandb.Video(video_path) 

        accelerator.print(f"collecting inferences from storage device and logging...") 
        for template_prompt in prompts_dataset.template_prompts:  
            # stores all the videos for this particular prompt on wandb  
            template_prompt_videos = {} 

            # collecting results of type 1 inference 
            prompts_dataset = prompts_dataset1 
            template_prompt_videos["type1"] = {} 
            for subject in sorted(prompts_dataset.subjects):  

                save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "type1")    

                subject_prompt = template_prompt.replace("SUBJECT", subject)   
                prompt_ = "_".join(subject_prompt.split()) 
                prompt_path = osp.join(save_path_global, prompt_) 
                img_names = os.listdir(prompt_path)   
                img_names = [img_name for img_name in img_names if img_name.find(f"jpg") != -1] 
                img_names = sorted(img_names) 

                assert "bnha" in subject 
                keyname = subject.replace(f"bnha", "pose+app") 

                template_prompt_videos["type1"][keyname] = [] 
                assert len(img_names) == prompts_dataset.num_samples, f"{len(img_names) = }, {prompts_dataset.num_samples = }" 
                for img_name in img_names: 
                    # prompt_path has a BUG 
                    # print(f"for {subject} i am using {prompt_path = } and {img_name = }") 
                    img_path = osp.join(prompt_path, img_name) 
                    got_image = False 
                    # while not got_image: 
                    #     try: 
                    #         img = Image.open(img_path) 
                    #         got_image = True 
                    #     except Exception as e: 
                    #         print(f"could not read the image, will try again, don't worry, just read and chill!") 
                    #         got_image = False 
                    #     if got_image: 
                    #         break 
                    img = Image.open(img_path) 
                    template_prompt_videos["type1"][keyname].append(img) 


            # collecting results of type 2 inference 
            # prompts_dataset = prompts_dataset2 
            # template_prompt_videos["type2"] = {} 
            # for subject in sorted(prompts_dataset.subjects):  

            #     save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "type2")    

            #     subject_prompt = template_prompt.replace("SUBJECT", subject)   
            #     prompt_ = "_".join(subject_prompt.split()) 
            #     prompt_path = osp.join(save_path_global, prompt_) 
            #     img_names = os.listdir(prompt_path)   
            #     img_names = [img_name for img_name in img_names if img_name.find(f"jpg") != -1] 
            #     img_names = sorted(img_names) 

            #     assert "bnha" in subject 
            #     keyname = subject.replace(f"bnha", "pose_only") 

            #     template_prompt_videos["type2"][keyname] = [] 
            #     assert len(img_names) == prompts_dataset.num_samples 
            #     for img_name in img_names: 
            #         # prompt_path has a BUG 
            #         # print(f"for {subject} i am using {prompt_path = } and {img_name = }") 
            #         img_path = osp.join(prompt_path, img_name) 
            #         got_image = False 
            #         # while not got_image: 
            #         #     try: 
            #         #         img = Image.open(img_path) 
            #         #         got_image = True 
            #         #     except Exception as e: 
            #         #         print(f"could not read the image, will try again, don't worry, just read and chill!") 
            #         #         got_image = False 
            #         #     if got_image: 
            #         #         break 
            #         img = Image.open(img_path) 
            #         template_prompt_videos["type2"][keyname].append(img) 


            # collecting results of type 3 inference 
            prompts_dataset = prompts_dataset3 
            template_prompt_videos["type3"] = {} 
            for subject in sorted(prompts_dataset.subjects):  

                save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "type3")    

                subject_prompt = template_prompt.replace("SUBJECT", subject)   
                prompt_ = "_".join(subject_prompt.split()) 
                prompt_path = osp.join(save_path_global, prompt_) 
                img_names = os.listdir(prompt_path)   
                img_names = [img_name for img_name in img_names if img_name.find(f"jpg") != -1] 
                img_names = sorted(img_names) 

                assert "bnha" in subject 
                keyname = subject.replace(f"bnha", "") 

                template_prompt_videos["type3"][keyname] = [] 
                assert len(img_names) == prompts_dataset.num_samples 
                for img_name in img_names: 
                    # prompt_path has a BUG 
                    # print(f"for {subject} i am using {prompt_path = } and {img_name = }") 
                    img_path = osp.join(prompt_path, img_name) 
                    got_image = False 
                    # while not got_image: 
                    #     try: 
                    #         img = Image.open(img_path) 
                    #         got_image = True 
                    #     except Exception as e: 
                    #         print(f"could not read the image, will try again, don't worry, just read and chill!") 
                    #         got_image = False 
                    #     if got_image: 
                    #         break 
                    img = Image.open(img_path) 
                    template_prompt_videos["type3"][keyname].append(img) 


            # concatenate all the images for this template prompt 
            all_concat_imgs = [] 
            save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}")  
            # for idx in range(prompts_dataset.num_samples): 
            #     images = [] 
            #     for subject in sorted(prompts_dataset.subjects): 
            #         images.append(template_prompt_videos[subject][idx]) 
            #     concat_img = create_image_with_captions(images, sorted(prompts_dataset.subjects))  
            #     all_concat_imgs.append(concat_img) 
            for idx in range(NUM_SAMPLES): 
                images = [] 
                captions = [] 
                for typename in list(template_prompt_videos.keys()): 
                    images_row = [] 
                    captions_row = [] 
                    for keyname in list(template_prompt_videos[typename].keys()): 
                        images_row.append(template_prompt_videos[typename][keyname][idx]) 
                        captions_row.append(keyname) 
                    images.append(images_row) 
                    captions.append(captions_row) 

                concat_img = create_image_with_captions(images, captions)  
                all_concat_imgs.append(concat_img) 


            template_prompt_ = "_".join(template_prompt.split()) 
            video_path = osp.join(save_path_global, template_prompt_ + ".gif")  
            create_gif(all_concat_imgs, video_path, 1) 
            if accelerator.is_main_process and args.wandb:  
                wandb_log_data[template_prompt] = wandb.Video(video_path) 


        return wandb_log_data  


        # sometimes the same index is passed to multiple gpus, therefore an explicit gathering has to be done to make sure no image has been "generated twice" 
        # accelerator.print(f"collecting outputs across processes...")  
        # generated_images = accelerator.gather(generated_images.unsqueeze(0)) 
        # gathered_generated_images = torch.zeros_like(generated_images[0]) 
        # generated_images = generated_images.permute(1, 0, 2, 3, 4)  
        # assert generated_images.shape[0] == encoder_hidden_states.shape[0] 
        # for idx in range(generated_images.shape[0]): 
        #     for gpu_idx in range(generated_images.shape[1]): 
        #         if torch.sum(generated_images[idx][gpu_idx]): 
        #             # this is a generated image 
        #             gathered_generated_images[idx] = generated_images[idx][gpu_idx] 
        # for idx in range(gathered_generated_images.shape[0]): 
        #     assert torch.sum(gathered_generated_images[idx]) 

        # generated_images = gathered_generated_images 
        # generated_images = generated_images.cpu().numpy() 
        # for idx in range(generated_images.shape[0]): 
        #     azimuth = idx // n_prompts_per_azimuth 
        #     prompt_idx = idx % n_prompts_per_azimuth 
        #     prompt = prompts_dataset.prompts[prompt_idx] 
        #     if prompt not in videos.keys(): 
        #         videos[prompt] = np.zeros((prompts_dataset.num_samples, 3, 512, 512)).astype(np.uint8)  
        #     videos[prompt][azimuth] = generated_images[idx].astype(np.uint8)  

        # accelerator.print(f"done!")  
        # vae = vae.to(torch.device(f"cpu")) 
        # return videos 


"""end Adobe CONFIDENTIAL"""

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--ada",
        action="store_true",
        help="whether training on ada, this would enable certain optimizations to reduce the memory usage", 
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to use wandb or not ",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="the wandb run name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="iisc",
        help="the wandb project name",
    )
    parser.add_argument(
        "--controlnet_prompts_file",
        type=str,
        default=None,
        required=True,
        help="path to the txt file containing prompts for controlnet augmentation",
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=None,
        required=True,
        help="root data directory",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--controlnet_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of from controlnet.",
    )
    
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        help="the directory where the intermediate visualizations and inferences are stored",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=1709, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--h_flip",
        action="store_true",
        help="Whether to hflip before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        help="Whether to train the unet",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--textual_inv",
        action="store_true",
        help="Whether to use textual inversion",
    )
    parser.add_argument(
        "--online_inference", 
        action="store_true", 
        help="whether to do online inference", 
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the inference dataloader.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="wandb log every ddp steps",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=None, 
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_mlp",
        type=float,
        default=None, 
        help="Initial learning rate for mlp (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_emb",
        type=float,
        default=None, 
        help="Initial learning rate for embedding (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_merger",
        type=float,
        default=None, 
        help="Initial learning rate for merger (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=-1,
        help="Number of steps for stage 1 training", 
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=100000,
        help="Number of steps for stage 2 training", 
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        required=True,
        help="the object name",
    )
    
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            logger.warning(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args


def main(args): 
    args.textual_inv = False 

    # subjects_ are the folders in the instance directory 
    subjects_ = os.listdir(args.instance_data_dir) 
    args.subjects = [" ".join(subject.split("_")) for subject in subjects_] 

    # defining the output directory to store checkpoints 
    args.output_dir = osp.join(args.output_dir, f"__{args.run_name}") 

    # storing the number of reference images per subject 
    args.n_ref_imgs = len(os.listdir(osp.join(args.instance_data_dir, subjects_[0]))) 

    # sanity check: for every subject there should be the same angles  
    # print(f"{subjects_ = }")
    # for subject_ in subjects_[:1]: 
    #     subject_path = osp.join(args.instance_data_dir, subject_) 
    #     files = os.listdir(subject_path) 
    #     angles = [float(file.replace(f".jpg", "")) for file in files] 
    #     angles = sorted(np.array(angles)) 

    # angles_ref = angles.copy()  
    # for subject_ in subjects_[1:]: 
    #     subject_path = osp.join(args.instance_data_dir, subject_) 
    #     files = os.listdir(subject_path) 
    #     angles = [float(file.replace(f".jpg", "")) for file in files] 
    #     angles = sorted(np.array(angles)) 
    #     assert np.allclose(angles, angles_ref) 

    # max train steps 
    args.max_train_steps = args.stage1_steps + args.stage2_steps + 1  

    # accelerator 
    accelerator = Accelerator(
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],  
        # find_unused_parameters=True, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # effective batch size should remain constant 
    assert accelerator.num_processes * args.train_batch_size == BS, f"{accelerator.num_processes = }, {args.train_batch_size = }" 

    # init wandb 
    if args.wandb and accelerator.is_main_process:
        wandb_config = vars(args) 
        wandb.login(key="6ab81b60046f7d7f6a7dca014a2fcaf4538ff14a") 
        if args.run_name is None: 
            wandb.init(project=args.project, config=wandb_config) 
        else:
            wandb.init(project=args.project, name=args.run_name, config=wandb_config) 
    
        

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # setting a different seed for each process to increase diversity in minibatch 
    set_seed(args.seed + accelerator.process_index) 

    # if args.with_prior_preservation:
    #     class_images_dir = Path(args.class_data_dir)
    #     if not class_images_dir.exists():
    #         class_images_dir.mkdir(parents=True)
    #     cur_class_images = len(list(class_images_dir.iterdir()))

    #     assert cur_class_images == args.num_class_images 

    # Handle the repository creation
    # handle the creation of output directory 
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet.requires_grad_(False)
    if args.train_unet: 
        unet_lora_params, _ = inject_trainable_lora(
            unet, r=args.lora_rank, loras=args.resume_unet
        )

    # for _up, _down in extract_lora_ups_down(unet):
    #     print("Before training: Unet First Layer lora up", _up.weight.data)
    #     print("Before training: Unet First Layer lora down", _down.weight.data)
    #     break

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # injecting trainable lora in text encoder 
    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )
        # for _up, _down in extract_lora_ups_down(
        #     text_encoder, target_replace_module=["CLIPAttention"]
        # ):
        #     print("Before training: text encoder First Layer lora up", _up.weight.data)
        #     print(
        #         "Before training: text encoder First Layer lora down", _down.weight.data
        #     )
        #     break

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()
    #     if args.train_text_encoder:
    #         text_encoder.gradient_checkpointing_enable()


    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    

    # params_to_optimize = (
    #     [
    #         {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
    #         {
    #             "params": itertools.chain(*text_encoder_lora_params),
    #             "lr": text_lr,
    #         },
    #     ]
    #     if args.train_text_encoder
    #     else itertools.chain(*unet_lora_params)
    # )
    # optimizer = optimizer_class(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    optimizers = {}  
    if args.train_unet: 
        optimizer_unet = optimizer_class(
            itertools.chain(*unet_lora_params), 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        # optimizers.append(optimizer_unet) 
        optimizers["unet"] = optimizer_unet 

    if args.train_text_encoder: 
        optimizer_text_encoder = optimizer_class(
            itertools.chain(*text_encoder_lora_params),  
            lr=args.learning_rate_text,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        # optimizers.append(optimizer_text_encoder) 
        optimizers["text_encoder"] = optimizer_text_encoder 

    if args.textual_inv: 
        # the appearance embeddings 
        bnha_embeds = {} 
        for subject in args.subjects:  
            # initializing using the subject's embedding in the pretrained CLIP text encoder 
            bnha_embeds[subject] = torch.clone(text_encoder.get_input_embeddings().weight[TOKEN2ID[subject]]).detach()  

        # initializing the AppearanceEmbeddings module using the embeddings 
        # bnha_embeds = AppearanceEmbeddings(bnha_embeds).to(accelerator.device) 

        # an optimizer for the appearance embeddings 
        optimizer_bnha = optimizer_class(
            bnha_embeds.parameters(),  
            lr=args.learning_rate_emb,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        # optimizers.append(optimizer_bnha) 
        optimizers["appearance"] = optimizer_bnha 


    pos_size = 6 
    continuous_word_model = continuous_word_mlp(input_size=pos_size, output_size=1024)
    optimizer_mlp = optimizer_class(
        continuous_word_model.parameters(),  
        lr=args.learning_rate_mlp,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizers.append(optimizer_mlp)  
    optimizers["contword"] = optimizer_mlp 


    # the merged token formulation 
    merger = MergedEmbedding()  
    # optimizer_merger = torch.optim.Adam(merger.parameters(), lr=args.learning_rate_merger)  
    optimizer_merger = optimizer_class(
        merger.parameters(),  
        lr=args.learning_rate_merger,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizers.append(optimizer_merger) 
    optimizers["merger"] = optimizer_merger 


    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # defining the dataset 
    train_dataset = DisentangleDataset(
        args=args,
        tokenizer=tokenizer, 
    )

    """
    ADOBE CONFIDENTIAL
    Copyright 2024 Adobe
    All Rights Reserved.
    NOTICE: All information contained herein is, and remains
    the property of Adobe and its suppliers, if any. The intellectual
    and technical concepts contained herein are proprietary to Adobe 
    and its suppliers and are protected by all applicable intellectual 
    property laws, including trade secret and copyright laws. 
    Dissemination of this information or reproduction of this material is 
    strictly forbidden unless prior written permission is obtained from Adobe.
    """
    def collate_fn(examples):
        is_controlnet = [example["controlnet"] for example in examples] 
        prompt_ids = [example["prompt_ids"] for example in examples] 
        prompts = [example["prompt"] for example in examples] 

        for example in examples: 
            assert TOKEN2ID["bnha"] in example["prompt_ids"] 
            assert TOKEN2ID[example["subject"]] not in example["prompt_ids"] 

        subjects = [example["subject"] for example in examples] 
        pixel_values = []
        for example in examples:
            pixel_values.append(example["img"])

        """Adding the scaler of the embedding into the batch"""
        # scalers = torch.Tensor([example["scaler"] for example in examples])
        scalers = [torch.tensor(example["scaler"]) for example in examples] 
        scalers = torch.stack(scalers, dim=0)  

        if args.with_prior_preservation:
            prompt_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_img"] for example in examples]
            prompts += [example["class_prompt"] for example in examples] 

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        prompt_ids = tokenizer.pad(
            {"input_ids": prompt_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        batch = {
            "prompts": prompts, 
            "prompt_ids": prompt_ids, 
            "pixel_values": pixel_values,
            "scalers": scalers,
            "subjects": subjects, 
            "controlnet": is_controlnet, 
        }

        return batch 
    """end Adobe CONFIDENTIAL"""

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=accelerator.num_processes * 2,
    )

    """
    ADOBE CONFIDENTIAL
    Copyright 2024 Adobe
    All Rights Reserved.
    NOTICE: All information contained herein is, and remains
    the property of Adobe and its suppliers, if any. The intellectual
    and technical concepts contained herein are proprietary to Adobe 
    and its suppliers and are protected by all applicable intellectual 
    property laws, including trade secret and copyright laws. 
    Dissemination of this information or reproduction of this material is 
    strictly forbidden unless prior written permission is obtained from Adobe.
    """
    # the mlp controlling the pose 
    # pos_size = 2
    # continuous_word_model = continuous_word_mlp(input_size=pos_size, output_size=1024)
    # continuous_word_optimizer = torch.optim.Adam(continuous_word_model.parameters(), lr=args.learning_rate_mlp) 
    # optimizers.append(continuous_word_optimizer) 
    # print("The current continuous MLP: {}".format(continuous_word_model))
    
    
    unet, text_encoder, merger, continuous_word_model, train_dataloader = accelerator.prepare(unet, text_encoder, merger, continuous_word_model, train_dataloader)  
    # optimizers_ = [] 
    optimizers_ = {} 
    for name, optimizer in optimizers.items(): 
        optimizer = accelerator.prepare(optimizer) 
        # optimizers_.append(optimizer) 
        optimizers_[name] = optimizer 
    if args.textual_inv: 
        bnha_embeds = accelerator.prepare(bnha_embeds) 
    optimizers = optimizers_  

    """End Adobe CONFIDENTIAL"""

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    """
    ADOBE CONFIDENTIAL
    Copyright 2024 Adobe
    All Rights Reserved.
    NOTICE: All information contained herein is, and remains
    the property of Adobe and its suppliers, if any. The intellectual
    and technical concepts contained herein are proprietary to Adobe 
    and its suppliers and are protected by all applicable intellectual 
    property laws, including trade secret and copyright laws. 
    Dissemination of this information or reproduction of this material is 
    strictly forbidden unless prior written permission is obtained from Adobe.
    """
    continuous_word_model.to(accelerator.device, dtype=weight_dtype)
    """End Adobe CONFIDENTIAL"""
    
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    global_step = 0
    ddp_step = 0

    
    if args.train_unet: 
        unet.train()
    """
    ADOBE CONFIDENTIAL
    Copyright 2024 Adobe
    All Rights Reserved.
    NOTICE: All information contained herein is, and remains
    the property of Adobe and its suppliers, if any. The intellectual
    and technical concepts contained herein are proprietary to Adobe 
    and its suppliers and are protected by all applicable intellectual 
    property laws, including trade secret and copyright laws. 
    Dissemination of this information or reproduction of this material is 
    strictly forbidden unless prior written permission is obtained from Adobe.
    """
    continuous_word_model.train()
    """End Adobe CONFIDENTIAL"""
    if args.train_text_encoder:
        text_encoder.train()

    if DEBUG: 
        if osp.exists(f"vis"): 
            shutil.rmtree(f"vis") 
        os.makedirs("vis")  

    input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight).detach()  

    # steps_per_angle = {} 

    for step, batch in enumerate(train_dataloader):
        if DEBUG: 
            assert torch.allclose(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight, input_embeddings) 
        # for batch_idx, angle in enumerate(batch["anagles"]): 
        #     if angle in steps_per_angle.keys(): 
        #         steps_per_angle[angle] += 1 
        #     else:
        #         steps_per_angle[angle] = 1 

        B = len(batch["scalers"])   
        wandb_log_data = {}
        force_wandb_log = False 
        # Convert images to latent space
        vae.to(accelerator.device, dtype=weight_dtype)

        if DEBUG: 
            for batch_idx, img_t in enumerate(batch["pixel_values"]): 
                img = (img_t * 0.5 + 0.5) * 255  
                img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8) 
                plt.imshow(img)  
                plt_title = f"{batch_idx = }\n{batch['prompts'][batch_idx] = }" 
                plt_title = "\n".join(textwrap.wrap(plt_title, width=60)) 
                plt.title(plt_title, fontsize=9)  
                plt.savefig(f"vis/{str(step).zfill(3)}_{str(batch_idx).zfill(3)}.jpg") 

        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * 0.18215
        if args.ada: 
            vae = vae.to(torch.device(f"cpu")) 

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        """
        ADOBE CONFIDENTIAL
        Copyright 2024 Adobe
        All Rights Reserved.
        NOTICE: All information contained herein is, and remains
        the property of Adobe and its suppliers, if any. The intellectual
        and technical concepts contained herein are proprietary to Adobe 
        and its suppliers and are protected by all applicable intellectual 
        property laws, including trade secret and copyright laws. 
        Dissemination of this information or reproduction of this material is 
        strictly forbidden unless prior written permission is obtained from Adobe.
        """

        # if we are in stage 2 of training, only then do we need to compute the pose embedding, otherwise it is zero 
        if global_step > args.stage1_steps: 
            progress_bar.set_description(f"stage 2: ")
            # p = torch.Tensor(batch["scalers"] / (2 * math.pi)) 
            p = torch.Tensor(batch["scalers"]) / torch.tensor([2 * math.pi, 2 * math.pi, 1.0]).to(accelerator.device)   
            # p = p.unsqueeze(-1) 
            assert p.shape == (B, 3) 
            p = p.repeat(1, 2) 
            assert torch.allclose(p[:, 0], p[:, 3]) 
            assert torch.allclose(p[:, 1], p[:, 4]) 
            assert torch.allclose(p[:, 2], p[:, 5]) 
            p[:, 0:3] = torch.sin(2 * torch.pi * p[:, 0:3]) 
            p[:, 3:6] = torch.cos(2 * torch.pi * p[:, 3:6]) 

            # getting the embeddings from the mlp
            mlp_emb = continuous_word_model(p) 

        else: 
            assert False 
            progress_bar.set_description(f"stage 1: ")
            mlp_emb = torch.zeros(B, 1024) 

        # appearance embeddings
        # textual inversion is used, then the embeddings are initialized with their classes  
        # else it is initialized with the default value for bnha 
        if args.textual_inv: 
            # bnha_emb = torch.stack([getattr(accelerator.unwrap_model(bnha_embeds), subject) for subject in batch["subjects"]])  
            # bnha_emb = torch.stack([bnha_embeds(subject) for subject in batch["subjects"]])  
            assert False 
            bnha_emb = [] 
            assert len(batch["controlnet"]) == B 
            for idx in range(B): 
                if batch["controlnet"][idx]: 
                    # if controlnet image, then replace the appearance embedding by the class embedding
                    bnha_emb.append(torch.clone(input_embeddings)[TOKEN2ID[batch["subjects"][idx]]])  
                else: 
                    # bnha_emb.append(bnha_embeds(batch["subjects"][idx])) 
                    bnha_emb.append(getattr(accelerator.unwrap_model(bnha_embeds), batch["subjects"][idx]))  
            bnha_emb = torch.stack(bnha_emb) 

        else: 
            # bnha_emb = torch.clone(input_embeddings).detach()[TOKEN2ID["bnha"]].unsqueeze(0).repeat(B, 1)  
            bnha_emb = [] 
            # bnha_emb = torch.clone(input_embeddings)[TOKEN2ID[]] 
            for idx in range(B): 
                bnha_emb.append(torch.clone(input_embeddings)[TOKEN2ID[batch["subjects"][idx]]].detach())  

            bnha_emb = torch.stack(bnha_emb) 

        # merging the appearance and pose embeddings 
        merged_emb = merger(mlp_emb, bnha_emb)  
        assert not torch.allclose(merged_emb, torch.zeros_like(merged_emb)) 
        merged_emb_norm = torch.linalg.norm(merged_emb)  
        assert merged_emb.shape[0] == B 

        # replacing the input embedding for sks by the mlp for each batch item, and then getting the output embeddings of the text encoder 
        # must run a for loop here, first changing the input embeddings of the text encoder for each 
        encoder_hidden_states = []
        input_ids, input_ids_prior = torch.chunk(batch["prompt_ids"], 2, dim=0) 

        for batch_idx, batch_item in enumerate(input_ids): 
            # replacing the text encoder input embeddings by the original ones and setting them to be COLD -- to enable replacement by a hot embedding  
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(torch.clone(input_embeddings), requires_grad=False)  

            assert TOKEN2ID["bnha"] in batch_item  
            assert TOKEN2ID[batch["subjects"][batch_idx]] not in batch_item  

            # performing the replacement on cold embeddings by a hot embedding -- allowed 
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_emb[batch_idx] 

            # appending to the encoder states 
            encoder_hidden_states.append(text_encoder(batch_item.unsqueeze(0))[0].squeeze()) 

        encoder_hidden_states = torch.stack(encoder_hidden_states)  

        # replacing the text encoder input embeddings by the original ones, this time setting them to be HOT, this will be useful in case we choose to do textual inversion 
        # here we are not cloning because these won't be stepped upon anyways, and this way we can save some memory also!  
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=True)   
        encoder_hidden_states_prior = text_encoder(input_ids_prior)[0] 
        assert encoder_hidden_states_prior.shape == encoder_hidden_states.shape 
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_prior], dim=0)

        """End Adobe CONFIDENTIAL"""


        # Predict the noise residual
        if args.ada: 
            torch.cuda.empty_cache() 
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        losses = [] 
        if args.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )
            losses.append(loss.detach()) 

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )
            losses.append(prior_loss.detach()) 

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.append(loss.detach()) 
            losses.append(torch.tensor(0.0).to(accelerator.device)) 

        losses = torch.stack(losses).to(accelerator.device) 

        if args.ada: 
            torch.cuda.empty_cache() 

        # checking if the parameters do require grads at least 
        if DEBUG: 
            for p in merger.parameters(): 
                assert p.requires_grad 


        accelerator.backward(loss)
        # everytime the continuous word mlp must receive gradients 
        if DEBUG: 
            with torch.no_grad(): 
                # checking that merger receives gradients 
                bad_merger_params = [(n, p) for (n, p) in merger.named_parameters() if p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))] 
                # assert len(bad_merger_params) == 0, f"{len(bad_merger_params) = }, {len(list(merger.parameters())) = }" 
                # print(f"{len(bad_merger_params) = }, {len(list(merger.parameters())) = }")  
                # for (n, p) in merger.named_parameters():  
                    # if (n, p) not in bad_merger_params:  
                        # print(f"{n, p = } in merger is NOT bad!")
                if global_step < args.stage1_steps: 
                    assert False 
                    assert len(bad_merger_params) < len(list(merger.parameters()))  
                # elif global_step > args.stage1_steps: 
                elif global_step > 1:   
                    # maafi for the first training step, because the merger is zero initialized!
                    assert len(bad_merger_params) == 0, f"{len(bad_merger_params) = }" 

                # checking that mlp receives gradients in stage 2 
                # print(f"merger does receive gradients!")
                bad_mlp_params = [(n, p) for (n, p) in continuous_word_model.named_parameters() if p.grad is None or torch.allclose(p.grad, torch.tensor(0.0).to(accelerator.device))]   
                # assert not ((len(bad_mlp_params) < len(list(continuous_word_model.parameters()))) ^ (global_step > args.stage1_steps))  
                # assert not ((len(bad_mlp_params) == 0) ^ (global_step > args.stage1_steps))  
                # maafi for the first training step, because the merger is zero initialized!
                if global_step > 1: 
                    # print(f"{len(bad_mlp_params) = }, {len(list(continuous_word_model.parameters())) = }")  
                    # assert len(bad_mlp_params) < len(list(continuous_word_model.parameters()))  
                    assert len(bad_mlp_params) == 0, f"{len(bad_mlp_params) = }" 
                    # print(f"mlp does receive gradients!")
                del bad_mlp_params 

                # checking for each appearance embedding whether it should receive gradients  
                # controlnet_subjects = [] 
                # ref_subjects = [] 
                # for idx in range(B): 
                #     if batch["controlnet"][idx]: 
                #         controlnet_subjects.append(batch["subjects"][idx]) 
                #     else: 
                #         ref_subjects.append(batch["subjects"][idx]) 
                # for subject in args.subjects: 
                #     if subject in ref_subjects: 
                #         # the appearance must receive some gradient 
                #         app_emb = getattr(accelerator.unwrap_model(bnha_embeds), subject)
                #         assert app_emb.grad is not None  
                #         assert not torch.allclose(app_emb.grad, torch.zeros(1024).to(accelerator.device))  
                #     else: 
                #         # the appearance must NOT have received any gradient 
                #         app_emb = getattr(accelerator.unwrap_model(bnha_embeds), subject)
                #         assert app_emb.grad is None or torch.allclose(app_emb.grad, torch.zeros(1024).to(accelerator.device))   
                assert torch.allclose(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight, input_embeddings) 
                    

                # checking whether the text encoder will receive gradients 
                if args.train_text_encoder: 
                    # some_grad_is_good = False  
                    # for p in list(text_encoder.parameters()):   
                    for p in list(itertools.chain(*text_encoder_lora_params)):    
                        if p.grad is None: 
                            continue 
                        # if not torch.allclose(p.grad, torch.zeros_like(p.grad)):   
                        #     some_grad_is_good = True 
                        assert not torch.allclose(p.grad, torch.zeros_like(p.grad))  
                    # assert some_grad_is_good 

                # checking whether the unet will receive gradients 
                if args.train_unet: 
                    # some_grad_is_good = False 
                    # for p in list(itertools.chain(*unet_lora_params)):    
                    for n, p in list(unet.named_parameters()):    
                        if p.grad is None: 
                            continue 
                        # print(f"{torch.zeros_like(p.grad) = }, {p.grad = }")
                        # print(f"something is not none also!")
                        if not torch.allclose(p.grad, torch.zeros_like(p.grad)):  
                            # print(f"{n = } has a gradient!")
                            some_grad_is_good = True 
                        else: 
                            # assert not torch.allclose(p.grad, torch.zeros_like(p.grad)) 
                            # print(f"{n = } DOES NOT HAVE GRADIENT...")
                            pass 
                    assert some_grad_is_good 


                # while debugging, go all controlnet, and then this assertion must pass 
                # check_bnha_params = [p for p in bnha_embeds.parameters() if p.grad is None or torch.allclose(p.grad, torch.tensor(0.0))] 
                # assert len(check_bnha_params) == len(list(bnha_embeds.parameters())) 
        """
        ADOBE CONFIDENTIAL
        Copyright 2024 Adobe
        All Rights Reserved.
        NOTICE: All information contained herein is, and remains
        the property of Adobe and its suppliers, if any. The intellectual
        and technical concepts contained herein are proprietary to Adobe 
        and its suppliers and are protected by all applicable intellectual 
        property laws, including trade secret and copyright laws. 
        Dissemination of this information or reproduction of this material is 
        strictly forbidden unless prior written permission is obtained from Adobe.
        """

        # since backward pass is done, the gradients would be ready! time to store grad norms!

        # calculate the gradient norm for each of the trainable parameters 
        if args.wandb and ((ddp_step + 1) % args.log_every == 0): 
            with torch.no_grad(): 
                all_grad_norms = []

                # mlp 
                mlp_grad_norm = [torch.linalg.norm(param.grad) for param in continuous_word_model.parameters() if param.grad is not None]
                if len(mlp_grad_norm) == 0:
                    mlp_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                else:
                    mlp_grad_norm = torch.mean(torch.stack(mlp_grad_norm)) 
                all_grad_norms.append(mlp_grad_norm) 


                # merger  
                merger_grad_norm = [torch.linalg.norm(param.grad) for param in merger.parameters() if param.grad is not None]
                if len(merger_grad_norm) == 0:
                    merger_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                else:
                    merger_grad_norm = torch.mean(torch.stack(merger_grad_norm)) 
                all_grad_norms.append(merger_grad_norm)  


                # unet 
                if args.train_unet: 
                    unet_grad_norm = [torch.linalg.norm(param.grad) for param in unet.parameters() if param.grad is not None]
                    if len(unet_grad_norm) == 0:
                        unet_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        unet_grad_norm = torch.mean(torch.stack(unet_grad_norm)) 
                    all_grad_norms.append(unet_grad_norm) 
                        
                # text encoder 
                if args.train_text_encoder: 
                    text_encoder_grad_norm = [torch.linalg.norm(param.grad) for param in text_encoder.parameters() if param.grad is not None]
                    if len(text_encoder_grad_norm) == 0:
                        text_encoder_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        text_encoder_grad_norm = torch.mean(torch.stack(text_encoder_grad_norm)) 
                    all_grad_norms.append(text_encoder_grad_norm) 
                
                # embedding  
                if args.textual_inv: 
                    bnha_grad_norm = [torch.linalg.norm(param.grad) for param in bnha_embeds.parameters() if param.grad is not None] 
                    if len(bnha_grad_norm) == 0: 
                        bnha_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else: 
                        bnha_grad_norm = torch.mean(torch.stack(bnha_grad_norm)) 
                    all_grad_norms.append(bnha_grad_norm) 


                # grad_norms would be in the order (if available): mlp, unet, text_encoder, embedding  
                # gathering all the norms at once to prevent excessive multi gpu communication 
                all_grad_norms = torch.stack(all_grad_norms).unsqueeze(0)  
                gathered_grad_norms = torch.mean(accelerator.gather(all_grad_norms), dim=0)  
                wandb_log_data["mlp_grad_norm"] = gathered_grad_norms[0] 
                wandb_log_data["merger_grad_norm"] = gathered_grad_norms[1]  
                curr = 2  
                while curr < len(gathered_grad_norms):  
                    if args.train_unet and ("unet_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["unet_grad_norm"] = gathered_grad_norms[curr]  

                    elif args.train_text_encoder and ("text_encoder_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["text_encoder_grad_norm"] = gathered_grad_norms[curr] 

                    elif args.textual_inv and ("bnha_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["bnha_grad_norm"] = gathered_grad_norms[curr] 
                    
                    else:
                        assert False 
                    curr += 1

        # gradient clipping 
        if accelerator.sync_gradients:
            params_to_clip = [] 
            parmas_to_clip = params_to_clip + list(itertools.chain(continuous_word_model.parameters())) + list(itertools.chain(merger.parameters()))  
            if args.train_unet: 
                params_to_clip = parmas_to_clip + list(itertools.chain(unet.parameters()))  
            if args.train_text_encoder: 
                params_to_clip = parmas_to_clip + list(itertools.chain(text_encoder.parameters()))  
            if args.textual_inv: 
                params_to_clip = params_to_clip + list(itertools.chain(bnha_embeds.parameters())) 
            # params_to_clip = (
            #     itertools.chain(unet.parameters(), text_encoder.parameters(), continuous_word_model.parameters())
            #     if args.train_text_encoder
            #     else itertools.chain(unet.parameters(), continuous_word_model.parameters())
            # )
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        # if DEBUG: 
            # with torch.no_grad(): 
                # merger_before = copy.deepcopy([p for p in merger.parameters()]) 
                # mlp_before = copy.deepcopy([p for p in continuous_word_model.parameters()])  
                # unet_before = copy.deepcopy([p for p in unet.parameters()]) 
                # text_encoder_before = copy.deepcopy([p for p in text_encoder.parameters()]) 
                # if args.textual_inv: 
                #     bnha_before = copy.deepcopy([p for p in bnha_embeds.parameters()]) 

        # lora_before = [torch.clone(p) for p in list(itertools.chain(*unet_lora_params))] 
        for name, optimizer in optimizers.items(): 
            optimizer.step() 
        # lora_after = [torch.clone(p) for p in list(itertools.chain(*unet_lora_params))] 
        # for p1, p2 in zip(lora_before, lora_after): 
        #     assert not torch.allclose(p1, p2) 
        #     print(f"unet_lora_params is changing!") 

        # calculating weight norms 
        if args.wandb and ((ddp_step + 1) % args.log_every == 0): 
            with torch.no_grad(): 
                all_norms = []

                # mlp 
                mlp_norm = [torch.linalg.norm(param) for param in continuous_word_model.parameters() if param.grad is not None]
                if len(mlp_norm) == 0:
                    mlp_norm = torch.tensor(0.0).to(accelerator.device) 
                else:
                    mlp_norm = torch.mean(torch.stack(mlp_norm)) 
                all_norms.append(mlp_norm) 


                # merger  
                merger_norm = [torch.linalg.norm(param) for param in merger.parameters() if param.grad is not None]
                if len(merger_norm) == 0:
                    merger_norm = torch.tensor(0.0).to(accelerator.device) 
                else:
                    merger_norm = torch.mean(torch.stack(merger_norm)) 
                all_norms.append(merger_norm) 

                # merged_embedding norm 
                all_norms.append(merged_emb_norm)  

                # unet 
                if args.train_unet: 
                    unet_norm = [torch.linalg.norm(param) for param in unet.parameters() if param.grad is not None]
                    if len(unet_norm) == 0:
                        unet_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        unet_norm = torch.mean(torch.stack(unet_norm)) 
                    all_norms.append(unet_norm) 
                        
                # text encoder 
                if args.train_text_encoder: 
                    text_encoder_norm = [torch.linalg.norm(param) for param in text_encoder.parameters() if param.grad is not None]
                    if len(text_encoder_norm) == 0:
                        text_encoder_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        text_encoder_norm = torch.mean(torch.stack(text_encoder_norm)) 
                    all_norms.append(text_encoder_norm) 
                
                # embedding  
                if args.textual_inv: 
                    bnha_norm = [torch.linalg.norm(param) for param in bnha_embeds.parameters() if param.grad is not None] 
                    if len(bnha_norm) == 0: 
                        bnha_norm = torch.tensor(0.0).to(accelerator.device) 
                    else: 
                        bnha_norm = torch.mean(torch.stack(bnha_norm)) 
                    all_norms.append(bnha_norm) 


                # grad_norms would be in the order (if available): mlp, unet, text_encoder, embedding  
                # gathering all the norms at once to prevent excessive multi gpu communication 
                all_norms = torch.stack(all_norms).unsqueeze(0)  
                gathered_norms = torch.mean(accelerator.gather(all_norms), dim=0)  
                wandb_log_data["mlp_norm"] = gathered_norms[0] 
                wandb_log_data["merger_norm"] = gathered_norms[1]  
                wandb_log_data["merged_emb_norm"] = gathered_norms[2] 
                curr = 3  
                while curr < len(gathered_norms):  
                    if args.train_unet and ("unet_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["unet_norm"] = gathered_norms[curr]  

                    elif args.train_text_encoder and ("text_encoder_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["text_encoder_norm"] = gathered_norms[curr] 

                    elif args.textual_inv and ("bnha_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["bnha_norm"] = gathered_norms[curr] 
                    
                    else:
                        assert False 
                    curr += 1

        # if DEBUG: 
        #     # checking that no parameter should be NaN 
        #     for p in merger.parameters(): 
        #         assert not torch.any(torch.isnan(p)) 
        #     for p in continuous_word_model.parameters(): 
        #         assert not torch.any(torch.isnan(p)) 
        #     for p in unet.parameters(): 
        #         assert not torch.any(torch.isnan(p)) 
        #     for p in text_encoder.parameters(): 
        #         assert not torch.any(torch.isnan(p)) 

        #     with torch.no_grad(): 
        #         merger_after = copy.deepcopy([p for p in merger.parameters()]) 
        #         mlp_after = copy.deepcopy([p for p in continuous_word_model.parameters()])  
        #         unet_after = copy.deepcopy([p for p in unet.parameters()]) 
        #         text_encoder_after = copy.deepcopy([p for p in text_encoder.parameters()]) 
        #         if args.textual_inv: 
        #             bnha_after = copy.deepcopy([p for p in bnha_embeds.parameters()]) 

        #         merger_after = [p1 - p2 for p1, p2 in zip(merger_before, merger_after)] 
        #         del merger_before 
        #         mlp_after = [p1 - p2 for p1, p2 in zip(mlp_before, mlp_after)] 
        #         del mlp_before 
        #         unet_after = [p1 - p2 for p1, p2 in zip(unet_before, unet_after)]  
        #         del unet_before 
        #         text_encoder_after = [p1 - p2 for p1, p2 in zip(text_encoder_before, text_encoder_after)]  
        #         del text_encoder_before
        #         if args.textual_inv: 
        #             bnha_after = [p1 - p2 for p1, p2 in zip(bnha_before, bnha_after)]   
        #             del bnha_before 

        #         change = False 
        #         for p_diff in merger_after: 
        #             if torch.sum(p_diff): 
        #                 change = True 
        #                 break 
        #         assert change 

        #         change = False 
        #         for p_diff in mlp_after:  
        #             if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
        #                 change = True 
        #                 break 
        #         assert not (change ^ (global_step > args.stage1_steps)), f"{change = }, {global_step = }, {args.stage1_steps = }" 

        #         change = False 
        #         for p_diff in unet_after:  
        #             if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
        #                 change = True 
        #                 break 
        #         assert not (change ^ args.train_unet)  

        #         change = False 
        #         for p_diff in text_encoder_after:  
        #             if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
        #                 change = True 
        #                 break 
        #         assert not (change ^ args.train_text_encoder)   
            
        #         if args.textual_inv and torch.sum(torch.tensor(batch["controlnet"])).item() < B:  
        #             change = False 
        #             for p_diff in bnha_after:  
        #                 if not torch.allclose(p_diff, torch.zeros_like(p_diff)):  
        #                     change = True 
        #                     break 
        #             assert not (change ^ args.textual_inv), f"{batch['controlnet'] = }" 


        progress_bar.update(accelerator.num_processes * args.train_batch_size) 

        # optimizer_unet.zero_grad()
        # optimizer_text_encoder.zero_grad()
        # continuous_word_optimizer.zero_grad()
        for name, optimizer in optimizers.items(): 
            optimizer.zero_grad() 

        """end Adobe CONFIDENTIAL"""

        # since we have stepped, time to log weight norms!

        global_step += accelerator.num_processes * args.train_batch_size  
        ddp_step += 1 

        if args.online_inference and len(VLOG_STEPS) > 0 and global_step >= VLOG_STEPS[0]:  
            step = VLOG_STEPS[0] 
            VLOG_STEPS.pop(0) 
            # if global_step <= args.stage1_steps:  
            #     use_sks = False 
            # else:
            #     use_sks = True 
            if DEBUG: 
                # unet_safe = copy.deepcopy(unet) 
                # text_encoder_safe = copy.deepcopy(text_encoder) 
                # mlp_safe = copy.deepcopy(continuous_word_model)  
                # merger_safe = copy.deepcopy(merger) 
                unet_params_safe = [torch.clone(p) for p in unet.parameters()] 
                text_encoder_params_safe = [torch.clone(p) for p in text_encoder.parameters()] 
                mlp_params_safe = [torch.clone(p) for p in continuous_word_model.parameters()] 
                merger_params_safe = [torch.clone(p) for p in merger.parameters()] 
                # bnha_embeds_safe = [torch.clone(p) for p in bnha_embeds.parameters()] 


            # if args.textual_inv and args.online_inference: 
            #     assert False 
            #     wandb_log_data = infer(args, step, wandb_log_data, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, merger, bnha_embeds) 
            #     force_wandb_log = True 
            #     set_seed(args.seed + accelerator.process_index) 
            # elif args.online_inference: 
            #     wandb_log_data = infer(args, step, wandb_log_data, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, merger) 
            #     force_wandb_log = True 
            #     set_seed(args.seed + accelerator.process_index) 

            if DEBUG: 
                for p_, p in zip(unet_params_safe, unet.parameters()): 
                    assert torch.allclose(p_, p) 
                for p_, p in zip(text_encoder_params_safe, text_encoder.parameters()):  
                    assert torch.allclose(p_, p) 
                for p_, p in zip(mlp_params_safe, continuous_word_model.parameters()): 
                    assert torch.allclose(p_, p) 
                for p_, p in zip(merger_params_safe, merger.parameters()):  
                    assert torch.allclose(p_, p) 
                # for p_, p in zip(bnha_embeds_safe, bnha_embeds.parameters()):  
                #     assert torch.allclose(p_, p) 


            # if accelerator.is_main_process: 
            #     for key, value in videos.items():  
                    # this weird transposing had to be done, because earlier was trying to save raw data, but that gives a lot of BT with wandb.Video 
                    # value = np.transpose(value, (0, 2, 3, 1)) 

                    # Get the frame size
                    # height, width, _ = value[0].shape

                    # # Create the video writer
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # video_writer = cv2.VideoWriter('temp.gif', fourcc, 1, (width, height))

                    # # Write the frames to the video
                    # for frame in value: 
                    #     video_writer.write(frame)

                    # # Release the video writer
                    # video_writer.release()

                    # prompt_foldername = "_".join(key.split()) 
                    # save_path = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}", prompt_foldername) 
                    # os.makedirs(save_path, exist_ok=True)  
                    # save_path = osp.join(save_path, prompt_foldername + ".gif") 
                    # create_gif(value, save_path, duration=1) 
                    # if args.wandb: 
                    #     wandb_log_data[key] = wandb.Video(save_path)    

                    # force_wandb_log = True 
                
                # trying to push raw data to wandb.Video, does not work properly 
                # if args.wandb: 
                #     for key, value in videos.items(): 
                #         wandb_log_data[key] = wandb.Video(value, fps=1)
                #     force_wandb_log = True 
                
                    # os.makedirs(osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}"), exist_ok=True)    

                # also saving the video locally! 
                # for key, value in videos.items(): 
                #     prompt_foldername = "_".join(key.split()) 
                #     os.makedirs(osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}", prompt_foldername), exist_ok=True) 
                #     for image_idx, image in enumerate(value):  
                #         # image would be present in cwh format 
                #         image = np.transpose(image, (1, 2, 0)) 
                #         image = Image.fromarray(image) 
                #         image.save(osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}", prompt_foldername, str(image_idx).zfill(3) + ".jpg"), exist_ok=True) 

                # TODO metrics computation 
                # generated_images = [] 
                # gt_images = [] 
                # prompts = [] 
                # ref_img_names = os.listdir(args.instance_data_dir) 
                # ref_angles = [float(img_name.split("_.jpg")[0]) for img_name in ref_img_names] 
                # ref_angles = np.array(ref_angles)   
                # for key, value in videos.items():  
                #     n_views = len(value) 
                #     for image_idx, image in enumerate(value): 
                #         print(f"{image_idx = }")
                #         image = np.transpose(image, (1, 2, 0)) 
                #         image = Image.fromarray(image) 
                #         generated_images.append(image) 
                #         prompts.append(key) 
                #         angle = (image_idx / n_views) * 2 * math.pi 
                #         print(f"{n_views = }")
                #         print(f"{angle = }")
                #         diffs = ref_angles - angle 
                #         best_match_idx = int(np.argmin(diffs))  
                #         print(f"{ref_img_names[best_match_idx] = }")
                #         gt_images.append(Image.open(osp.join(args.instance_data_dir, ref_img_names[best_match_idx]))) 
                #         sys.exit(0) 
                

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # if args.save_steps and global_step - last_save >= args.save_steps:
            if len(SAVE_STEPS) > 0 and global_step >= SAVE_STEPS[0]: 
                save_step = SAVE_STEPS[0] 
                SAVE_STEPS.pop(0) 
                if accelerator.is_main_process:
                    # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                    # it, the models will be unwrapped, and when they are then used for further training,
                    # we will crash. pass this, but only to newer versions of accelerate. fixes
                    # https://github.com/huggingface/diffusers/issues/1566
                    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                        inspect.signature(
                            accelerator.unwrap_model
                        ).parameters.keys()
                    )
                    extra_args = (
                        {"keep_fp32_wrapper": True}
                        if accepts_keep_fp32_wrapper
                        else {}
                    )
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet, **extra_args),
                        text_encoder=accelerator.unwrap_model(
                            text_encoder, **extra_args
                        ),
                        revision=args.revision,
                    )

                    # filename_unet = (
                    #     f"{args.output_dir}/lora_weight_{global_step}.pt"
                    # )
                    # filename_text_encoder = f"{args.output_dir}/lora_weight_{global_step}.text_encoder.pt"
                    # print(f"save weights {filename_unet}, {filename_text_encoder}")
                    # save_lora_weight(pipeline.unet, filename_unet)
                    
                    # if args.output_format == "safe" or args.output_format == "both":

                    # this is for the whole model along with the optimizers 
                    training_state = {} 
                    # for name, optimizer in optimizers.items(): 
                    #     training_state[name] = {} 
                    #     training_state[name]["optimizer"] = optimizer.state_dict() 
                    #     training_state[name]["parameters"] =  

                    training_state["appearance"] = {} 
                    training_state["contword"] = {} 
                    training_state["merger"] = {} 
                    training_state["text_encoder"] = {} 
                    training_state["unet"] = {} 

                    training_state["merger"]["optimizer"] = optimizers["merger"].state_dict() 
                    training_state["merger"]["model"] = accelerator.unwrap_model(merger).state_dict() 

                    training_state["contword"]["optimizer"] = optimizers["contword"].state_dict() 
                    training_state["contword"]["model"] = accelerator.unwrap_model(continuous_word_model).state_dict() 

                    if args.textual_inv: 
                        training_state["appearance"]["optimizer"] = optimizers["appearance"].state_dict() 
                        training_state["appearance"]["model"] = accelerator.unwrap_model(bnha_embeds).state_dict()  

                    if args.train_unet: 
                        training_state["unet"]["optimizer"] = optimizers["unet"].state_dict() 
                        training_state["unet"]["model"] = args.pretrained_model_name_or_path  
                        training_state["unet"]["lora"] = list(itertools.chain(*unet_lora_params)) 

                    if args.train_text_encoder: 
                        training_state["text_encoder"]["optimizer"] = optimizers["text_encoder"].state_dict() 
                        training_state["text_encoder"]["model"] = args.pretrained_model_name_or_path  
                        training_state["text_encoder"]["lora"] = list(itertools.chain(*text_encoder_lora_params)) 

                    save_dir = osp.join(args.output_dir, f"training_state_{global_step}.pth")
                    torch.save(training_state, save_dir)   

                    # this is for saving the safeloras 
                    loras = {}
                    if args.train_unet: 
                        loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})

                    print("Cross Attention is also updated!")

                    # """ If updating only cross attention """
                    # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})

                    if args.train_text_encoder:
                        loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

                    if loras != {}: 
                        save_safeloras(loras, f"{args.output_dir}/lora_weight_{global_step}.safetensors")
                    
                    """
                    ADOBE CONFIDENTIAL
                    Copyright 2024 Adobe
                    All Rights Reserved.
                    NOTICE: All information contained herein is, and remains
                    the property of Adobe and its suppliers, if any. The intellectual
                    and technical concepts contained herein are proprietary to Adobe 
                    and its suppliers and are protected by all applicable intellectual 
                    property laws, including trade secret and copyright laws. 
                    Dissemination of this information or reproduction of this material is 
                    strictly forbidden unless prior written permission is obtained from Adobe.
                    """
                    # torch.save(continuous_word_model.state_dict(), f"{args.output_dir}/mlp_{global_step}.pt")
                    """End Adobe CONFIDENTIAL"""
                    
                    
                    # if args.train_text_encoder:
                    #     save_lora_weight(
                    #         pipeline.text_encoder,
                    #         filename_text_encoder,
                    #         target_replace_module=["CLIPAttention"],
                    #     )

                    # for _up, _down in extract_lora_ups_down(pipeline.unet):
                    #     print(
                    #         "First Unet Layer's Up Weight is now : ",
                    #         _up.weight.data,
                    #     )
                    #     print(
                    #         "First Unet Layer's Down Weight is now : ",
                    #         _down.weight.data,
                    #     )
                    #     break
                    # if args.train_text_encoder:
                    #     for _up, _down in extract_lora_ups_down(
                    #         pipeline.text_encoder,
                    #         target_replace_module=["CLIPAttention"],
                    #     ):
                    #         print(
                    #             "First Text Encoder Layer's Up Weight is now : ",
                    #             _up.weight.data,
                    #         )
                    #         print(
                    #             "First Text Encoder Layer's Down Weight is now : ",
                    #             _down.weight.data,
                    #         )
                    #         break



        loss = loss.detach()
        gathered_loss = torch.mean(accelerator.gather(loss), 0)
        # on gathering the list of losses, the shape will be (G, 2) if there are 2 losses 
        # mean along the zeroth dimension would give the actual losses 
        losses = losses.unsqueeze(0) 
        gathered_losses = torch.mean(accelerator.gather(losses), dim=0) 
        if args.wandb and ddp_step % args.log_every == 0:
            # wandb_log_data["loss"] = gathered_loss
            wandb_log_data["corrected_mse_loss"] = gathered_losses[0]   
            wandb_log_data["corrected_prior_loss"] = gathered_losses[1] 

        if args.wandb: 
            # finally logging!
            if accelerator.is_main_process and (force_wandb_log or ddp_step % args.log_every == 0): 
                for step in range(global_step - BS, global_step - 1): 
                    wandb.log({
                        "global_step": step, 
                    })
                wandb_log_data["global_step"] = global_step 
                wandb.log(wandb_log_data) 

            # hack to make sure that the wandb step and the global_step are in sync 
            elif accelerator.is_main_process: 
                for step in range(global_step - BS, global_step): 
                    wandb.log({
                        "global_step": step, 
                    })

        logs = {"loss": gathered_loss.item()} 

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # TODO do this taking care of the multi gpu thing 
    # print(f"{steps_per_angle = }")
    # steps_per_angle = list(steps_per_angle.values())  
    # steps_per_angle = np.array(steps_per_angle).astype(np.int32) 
    # plt.bar(range(len(steps_per_angle)), steps_per_angle, color="green") 

    wandb.finish() 

    # if accelerator.is_main_process: 
        # print(f"removing the intermediate GIFs...") 
        # all_filenames = [os.listdir(".")] 
        # all_gifs = [filename for filename in all_filenames if filename.find(f".gif") != -1] 
        # [shutil.rmtree(file) for file in all_gifs] 

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\nLora TRAINING DONE!\n\n")

        if args.output_format == "pt" or args.output_format == "both":
            save_lora_weight(pipeline.unet, args.output_dir + "/lora_weight.pt")
            if args.train_text_encoder:
                save_lora_weight(
                    pipeline.text_encoder,
                    args.output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )

        if args.output_format == "safe" or args.output_format == "both":
            loras = {}
            if args.train_unet: 
                loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})
            
            print("Cross Attention is also updated!")
            
            # """ If updating only cross attention """
            # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})
            
            if args.train_text_encoder:
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            if loras != {}: 
                save_safeloras(loras, args.output_dir + "/lora_weight.safetensors")
            """
            ADOBE CONFIDENTIAL
            Copyright 2024 Adobe
            All Rights Reserved.
            NOTICE: All information contained herein is, and remains
            the property of Adobe and its suppliers, if any. The intellectual
            and technical concepts contained herein are proprietary to Adobe 
            and its suppliers and are protected by all applicable intellectual 
            property laws, including trade secret and copyright laws. 
            Dissemination of this information or reproduction of this material is 
            strictly forbidden unless prior written permission is obtained from Adobe.
            """
            torch.save(continuous_word_model.state_dict(), args.output_dir + "/continuous_word_mlp.pt")
            """end Adobe CONFIDENTIAL"""

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training",
                blocking=False,
                auto_lfs_prune=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    controlnet_prompts = []
    prompts_file = open(args.controlnet_prompts_file)
    for line in prompts_file.readlines():
        prompt = str(line)
        prompt = "a photo of " + prompt 
        controlnet_prompts.append(prompt)
    args.controlnet_prompts = controlnet_prompts 
    main(args)