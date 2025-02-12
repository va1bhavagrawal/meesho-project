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
VLOG_STEPS = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000]
SAVE_STEPS = copy.deepcopy(VLOG_STEPS) 
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
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms

from pathlib import Path

import random
import re

from continuous_word_mlp import continuous_word_mlp, AppearanceEmbeddings, MergedEmbedding  
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb 

from datasets import PromptDataset  


class EncoderStatesDataset(Dataset): 
    def __init__(self, encoder_states, save_paths): 
        assert len(encoder_states) == len(save_paths) 
        self.encoder_states = encoder_states 
        self.save_paths = save_paths 
        for save_path in self.save_paths: 
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
    def __init__(self, gif_name, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, bnha_embeds=None, bs=4):   
        self.accelerator = accelerator 
        self.unet = unet 
        self.text_encoder = text_encoder  
        self.scheduler = scheduler 
        self.vae = vae 
        self.mlp = mlp 
        self.merger = merger 
        self.bnha_embeds = bnha_embeds 
        self.bs = bs 
        self.vis_dir = "./tmp"  
        if osp.exists(self.vis_dir) and self.accelerator.is_main_process: 
            shutil.rmtree(f"{self.vis_dir}") 
        self.tokenizer = tokenizer 
        self.gif_name = gif_name  

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
        latents = torch.randn(B, 4, 64, 64).to(self.accelerator.device)  
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
        save_path_global = osp.join(self.vis_dir)  
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


    def do_it(self, prompt, subjects, n_samples, app_embs, pose_embs):    
        with torch.no_grad(): 
            encoder_states = torch.zeros((n_samples * 1, 77, 1024)) 
            normalized_azimuths = np.arange(n_samples) / n_samples 
            
            save_paths = [] 
            for azimuth_idx, azimuth in enumerate(normalized_azimuths):  
                self.accelerator.print(f"{azimuth = }")
                sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(azimuth)), torch.cos(2 * torch.pi * torch.tensor(azimuth))]).to(accelerator.device)  
                mlp_embs = mlp(sincos.unsqueeze(0))   
                subject_prompts = [] 
                bnha_embs = [] 
                sks_embs = [] 
                # for subject_idx, subject in enumerate(subjects):  
                assert f"SUBJECT" in prompt 
                # subject_ = "_".join(subject.split()) 

                # subject_without_bnha = subject.replace("bnha", "").strip()  
                first_subject = subjects[0] 
                second_subject = subjects[1] 
                subject_prompt = prompt.replace(f"SUBJECT", f"sks {first_subject} and bnha {second_subject}")  
                self.accelerator.print(f"{subject_prompt = }")

                bnha_embs.append(getattr(self.accelerator.unwrap_model(bnha_embeds), first_subject))  
                sks_embs.append(getattr(self.accelerator.unwrap_model(bnha_embeds), second_subject)) 

                first_subject_ = "_".join(first_subject.split()) 
                second_subject_ = "_".join(second_subject.split()) 
                subject_ = f"{first_subject_}_and_{second_subject_}" 

                subject_prompts.append(subject_prompt) 
                save_paths.append(osp.join(self.vis_dir, subject_, f"{str(azimuth_idx).zfill(3)}.jpg")) 
                    
                bnha_embs = torch.stack((bnha_embs), 0) 
                sks_embs = torch.stack((sks_embs), 0) 
                merged_embs = merger(mlp_embs, bnha_embs) 
                merged_embs2 = merger(mlp_embs, sks_embs) 

                assert len(subject_prompts) == len(merged_embs) 
                assert len(subject_prompts) == len(merged_embs2) 
                assert len(subject_prompts) == 1 

                # assert len(save_paths) == len(merged_embs) 
                # for i in range(merged_embs.shape[0]): 
                tokens = self.tokenizer(
                    subject_prompt, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids 

                self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = merged_embs[0]  
                self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = merged_embs2[0]  

                assert TOKEN2ID["bnha"] in tokens 
                assert TOKEN2ID["sks"] in tokens 

                text_encoder_outputs = self.text_encoder(tokens.to(self.accelerator.device))[0].squeeze()   
                encoder_states[azimuth_idx] = text_encoder_outputs  


            # for subject in subjects: 
            #     subject_ = "_".join(subject.split()) 
            #     os.makedirs(osp.join(self.vis_dir, subject_)) 
            #     for azimuth_idx in range(n_samples): 
            #         save_path = osp.join(self.vis_dir, subject_, f"{azimuth_idx.zfill(3)}.jpg") 
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

            if accelerator.is_main_process: 
                collect_generated_images_for_single_subject([f"{first_subject_}_and_{second_subject_}"], self.vis_dir, prompt, "pose+app", self.gif_name)  




if __name__ == "__main__": 
    with torch.no_grad(): 
        accelerator = Accelerator(
            # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],  
            # find_unused_parameters=True, 
            # gradient_accumulation_steps=args.gradient_accumulation_steps,
            # mixed_precision=args.mixed_precision,
        )

        ckpt_path = "../ckpts/multiobject/__poseapp_imp_infer/training_state_70000.pth"  
        basename_ckpt = osp.basename(ckpt_path) 
        lora_path = ckpt_path.replace(basename_ckpt, f"lora_weight_70000.safetensors") 

        accelerator.print(f"{ckpt_path = }")
        accelerator.print(f"{lora_path = }")

        # lora_weights = torch.load(lora_path) 
        # print(f"{type(lora_weights) = }")
        # print(f"{lora_weights.keys() = }")
        # sys.exit(0) 

        assert osp.exists(ckpt_path) 
        assert osp.exists(lora_path), f"{lora_path = }" 
        accelerator.print(f"loading stable diffusion checkpoint...") 
        # pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to(accelerator.device)  
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(accelerator.device)   
        
        accelerator.print(f"loading finetuned model checkpoint...") 
        training_state = torch.load(ckpt_path) 
        # print(f"{training_state['unet']['lora'] = }")
        subjects_ = os.listdir("../training_data_vaibhav/prior_imgs_multiobject/") 
        subjects = [" ".join(subject.split("_")) for subject in subjects_] 

        accelerator.print(f"preparing appearance embeddings...")
        app_embs = {} 
        for subject in subjects:  
            # initializing using the subject's embedding in the pretrained CLIP text encoder 
            app_embs[subject] = torch.clone(pipeline.text_encoder.get_input_embeddings().weight[TOKEN2ID[subject]]).detach()  

        # initializing the AppearanceEmbeddings module using the embeddings 
        bnha_embeds = AppearanceEmbeddings(app_embs).to(accelerator.device) 
        retval = bnha_embeds.load_state_dict(training_state["appearance"]["model"]) 
        # retval = AppearanceEmbeddings(app_embs).to(accelerator.device).load_state_dict(training_state["appearance"]["model"])  
        # print(f"{good = }")
        # print(f"{bad = }")
        # print(f"{retval = }")
        # for name, p in bnha_embeds.named_parameters(): 
        #     print(f"{name = }")

        accelerator.print(f"preparing merger...")
        merger = MergedEmbedding().to(accelerator.device) 
        merger.load_state_dict(training_state["merger"]["model"]) 

        mlp = continuous_word_mlp(2, 1024).to(accelerator.device) 
        mlp.load_state_dict(training_state["contword"]["model"]) 

        # print(f"preparing the unet...") 
        # unet_before = [torch.clone(p) for p in pipeline.unet.parameters()] 
        # unet_lora_params, _ = inject_trainable_lora(
        #     pipeline.unet, r=4 
        # )

        # print(f"{training_state['unet']['lora'] = }")
        # print(f"{len(training_state['unet']['lora']) = }")
        # print(f"{len(list(itertools.chain(*unet_lora_params))) = }")
        # for idx, param in enumerate(list(itertools.chain(*unet_lora_params))):  
        #     param.fill_(training_state["unet"]["lora"][idx])   
        # unet_after = [torch.clone(p) for p in pipeline.unet.parameters()]  
        # for p1, p2 in zip(unet_before, unet_after): 
        #     assert not torch.allclose(p1, p2) 
        
        accelerator.print(f"patching the pipe with the lora weights...") 
        # patch_pipe(
        #     pipeline,
        #     lora_path, 
        #     patch_text="text_encoder" in training_state.keys(),
        #     patch_ti=False,
        #     patch_unet="unet" in training_state.keys(),
        # )
        # lora_weights = load_file(lora_path) 
        # print(f"{lora_weights = }")
        unet_before = copy.deepcopy([torch.clone(p) for p in pipeline.unet.parameters()])  
        text_encoder_before = [torch.clone(p) for p in pipeline.text_encoder.parameters()] 
        patch_pipe(
            pipeline, 
            lora_path, 
            patch_text=True, 
            patch_ti=True, 
            patch_unet=True, 
        )
        unet_after = copy.deepcopy([torch.clone(p) for p in pipeline.unet.parameters()])  
        text_encoder_after = [torch.clone(p) for p in pipeline.text_encoder.parameters()] 
        change = False 
        for p1, p2 in zip(unet_before, unet_after): 
            if not torch.allclose(p1, p2): 
                change = True 
                break 
        assert change 
        # for p1, p2 in zip(text_encoder_before, text_encoder_after): 
        #     if not torch.allclose(p1, p2): 
        #         change = True 
        #         break 
        # assert change 

        infer = Infer( 
            "output.gif",  
            accelerator, 
            pipeline.unet, 
            pipeline.scheduler, 
            pipeline.vae, 
            pipeline.text_encoder, 
            pipeline.tokenizer, 
            mlp,  
            merger, 
            bnha_embeds,  
        )

        prompt = "a photo of SUBJECT side by side" 
        subjects = [
            # "bnha pickup truck and sks motorbike" 
            "cat", 
            "horse", 
        ]
        n_samples = 18   

        infer.do_it(
            prompt,  
            subjects, 
            n_samples, 
            [], 
            [], 
        )