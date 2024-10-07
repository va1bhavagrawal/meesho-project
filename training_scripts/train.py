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
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np 
from io import BytesIO

from utils import * 

import matplotlib.pyplot as plt 
import textwrap 
from infer_online import Infer  
from distutils.util import strtobool 

import pickle 

from custom_attention_processor import patch_custom_attention 

# from metrics import MetricEvaluator 


# TOKEN2ID = {
#     "sks": 48136, 
#     "bnha": 49336,  
#     "pickup truck": 4629, # using the token for "truck" instead  
#     "bus": 2840, 
#     "cat": 2368, 
#     "giraffe": 22826, 
#     "horse": 4558,
#     "lion": 5567,  
#     "elephant": 10299,   
#     "jeep": 11286,  
#     "motorbike": 33341,  
#     "bicycle": 11652, 
#     "tractor": 14607,  
#     "truck": 4629,  
#     "zebra": 22548,  
#     "sedan": 24237, 
#     "hen": 8047, 
#     "shoe": 7342, 
#     "dog": 1929, 
# }
from infer_online import TOKEN2ID, UNIQUE_TOKENS 

DEBUG = False  
PRINT_STUFF = False  
BS = 4    
# SAVE_STEPS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000] 
# VLOG_STEPS = [4, 50, 100, 200, 500, 1000]   
# VLOG_STEPS = [50000, 
# VLOG_STEPS = []
# for vlog_step in range(0, 400000, 25000): 
#     VLOG_STEPS = VLOG_STEPS + [vlog_step]  
# VLOG_STEPS = sorted(VLOG_STEPS) 
VLOG_STEPS_GAP = 33000  
SAVE_STEPS_GAP = 5000  
# SAVE_STEPS = copy.deepcopy(VLOG_STEPS) 
# SAVE_STEPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]  
# SAVE_STEPS = [500, 1000, 5000]  
# for save_step in range(0, 400000, 10000): 
#     SAVE_STEPS = SAVE_STEPS + [save_step] 
# SAVE_STEPS = sorted(SAVE_STEPS) 


NUM_SAMPLES = 4  

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

from lora_diffusion import (
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

from continuous_word_mlp import continuous_word_mlp, AppearanceEmbeddings, MergedEmbedding, PoseEmbedding, PoseLocationEmbedding  
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb 

# from datasets import PromptDataset  

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

# def create_gif(images, save_path, duration=1):
#     """
#     Convert a sequence of NumPy array images to a GIF.
    
#     Args:
#         images (list): A list of NumPy array images.
#         fps (int): The frames per second of the GIF (default is 1).
#         loop (int): The number of times the animation should loop (0 means loop indefinitely) (default is 0).
#     """
#     frames = []
#     for img in images:
#         # Convert NumPy array to PIL Image
#         # img_pil = Image.fromarray(img.astype(np.uint8))
#         img_pil = img 
#         # Append to frames list
#         frames.append(img_pil)
    
#     # Save frames to a BytesIO object
#     # bytes_io = BytesIO()
#     # frames[0].save(bytes_io, save_all=True, append_images=frames[1:], duration=1000/fps, loop=loop, 
#                 #    disposal=2, optimize=True, subrectangles=True)
#     frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=int(duration * 1000))
    
#     # gif_bytes = bytes_io.getvalue()
#     # with open("temp.gif", "wb") as f:
#     #     f.write(gif_bytes)
#     # return gif_bytes 
#     return 


def infer(args, step_number, wandb_log_data, accelerator, unet, scheduler, vae, text_encoder, mlp, merger, bnha_embeds, input_embeddings_safe, max_subjects_per_example): 
    # making a copy of the text encoder because this will be changed during the inference process 
    MAX_SUBJECTS_PER_EXAMPLE = max_subjects_per_example 
    # print(f"passing {MAX_SUBJECTS_PER_EXAMPLE = }")
    with torch.no_grad(): 

        retval = patch_custom_attention(accelerator.unwrap_model(unet), store_attn=False, across_timesteps=False, store_loss=False)  

        if accelerator.is_main_process: 
            os.makedirs(args.vis_dir, exist_ok=True) 
            if osp.exists(osp.join(args.vis_dir, f"outputs_{step_number}")): 
                shutil.rmtree(osp.join(args.vis_dir, f"outputs_{step_number}")) 
            os.mkdir(osp.join(args.vis_dir, f"outputs_{step_number}")) 
        accelerator.wait_for_everyone() 

        tokenizer = CLIPTokenizer.from_pretrained( 
            args.pretrained_model_name_or_path,
            subfolder="tokenizer", 
        ) 
        tmp_dir = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "tmp")    
        # if args.textual_inv: 
        #     assert bnha_embeds is not None 
        #     infer = Infer(args.seed, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, tmp_dir, args.text_encoder_bypass, bnha_embeds, bs=args.inference_batch_size)    
        # else: 
        # PERFORMING ONLY CLASS INFERENCE HERE!
        assert bnha_embeds is None 
        infer = Infer(args.merged_emb_dim, accelerator, unet, scheduler, vae, text_encoder, tokenizer, mlp, merger, tmp_dir=tmp_dir, bnha_embeds=None, store_attn=False, bs=args.inference_batch_size)  


        prompt = "a photo of PLACEHOLDER in the streets of Venice with the sun setting in the background"   
        if accelerator.is_main_process: 
            if osp.exists("best_latents.pt"): 
                os.remove("best_latents.pt")  
            seed = random.randint(0, 170904) 
            with open(f"seed.pkl", "wb") as f: 
                pickle.dump(seed, f) 
            # set_seed(seed) 
            latents = torch.randn(1, 4, 64, 64)  
            with open(f"best_latents.pt", "wb") as f: 
                torch.save(latents, f) 
        accelerator.wait_for_everyone() 
        if not accelerator.is_main_process: 
            with open("seed.pkl", "rb") as f: 
                seed = pickle.load(f) 
        accelerator.wait_for_everyone() 
        gif_path = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "_".join(prompt.split()).strip() + ".gif")   
        subjects = [
            [
                {
                    "subject": "pickup truck", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.3, 
                    "y": 0.6,  
                }, 
                {
                    "subject": "jeep", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),  
                    "x": 0.7, 
                    "y": 0.7,  
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE],  
            [
                {
                    "subject": "dog", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.4, 
                    "y": 0.6, 
                }, 
                {
                    "subject": "tractor", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),  
                    "x": 0.8, 
                    "y": 0.9, 
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE],  
            [
                {
                    "subject": "sedan", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.3, 
                    "y": 0.5, 
                }, 
                {
                    "subject": "bus", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.7, 
                    "y": 0.7,   
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE], 
        ] 

        infer.do_it(seed, gif_path, prompt, subjects, args=args.__dict__)    
        assert osp.exists(gif_path) 
        wandb_log_data[prompt] = wandb.Video(gif_path)  
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = nn.Parameter(torch.clone(input_embeddings_safe), requires_grad=False) 


        prompt = "a photo of PLACEHOLDER in front of a serene waterfall, featuring a lot of greenery and rainy skies, and stones scattered around everywhere"    
        if accelerator.is_main_process: 
            if osp.exists("best_latents.pt"): 
                os.remove("best_latents.pt")  
            seed = random.randint(0, 170904) 
            with open(f"seed.pkl", "wb") as f: 
                pickle.dump(seed, f) 
            # set_seed(seed) 
            latents = torch.randn(1, 4, 64, 64)  
            with open(f"best_latents.pt", "wb") as f: 
                torch.save(latents, f) 
        accelerator.wait_for_everyone() 
        if not accelerator.is_main_process: 
            with open("seed.pkl", "rb") as f: 
                seed = pickle.load(f) 
        accelerator.wait_for_everyone() 
        gif_path = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}", "_".join(prompt.split()).strip() + ".gif")   
        subjects = [
            [
                {
                    "subject": "suv", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.3, 
                    "y": 0.6,  
                }, 
                {
                    "subject": "jeep", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),  
                    "x": 0.7, 
                    "y": 0.7,  
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE],  
            [
                {
                    "subject": "elephant", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.4, 
                    "y": 0.6, 
                }, 
                {
                    "subject": "jeep", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),  
                    "x": 0.8, 
                    "y": 0.9, 
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE],  
            [
                {
                    "subject": "horse", 
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.3, 
                    "y": 0.5, 
                }, 
                {
                    "subject": "bus", 
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES),   
                    "x": 0.7, 
                    "y": 0.7,   
                }
            ][:MAX_SUBJECTS_PER_EXAMPLE], 
        ] 

        infer.do_it(seed, gif_path, prompt, subjects, args=args.__dict__)     
        assert osp.exists(gif_path) 
        wandb_log_data[prompt] = wandb.Video(gif_path)  
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = nn.Parameter(torch.clone(input_embeddings_safe), requires_grad=False) 
        
        return wandb_log_data 


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
        "--instance_data_dir_1subject",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_2subjects",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images of single subject",
    )
    parser.add_argument(
        "--controlnet_data_dir_1subject",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of from controlnet.",
    )
    parser.add_argument(
        "--controlnet_data_dir_2subjects",
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
        "--include_class_in_prompt",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to include the class name in the prompt, aka the subject in prompt approach!",
    )
    parser.add_argument(
        "--pose_only_embedding", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to only use pose in the merged embedding?", 
    )
    parser.add_argument(
        "--normalize_merged_embedding", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to normalize the merged embedding, would normalize even when include_class_in_prompt is True", 
    )
    parser.add_argument(
        "--use_location_conditioning", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use location conditioning", 
    )
    parser.add_argument(
        "--learn_pose", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="learn the pose embedding",  
    )
    parser.add_argument(
        "--attn_bbox_from_class_mean", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the class attention map's mean to make a bounding box attention mask for both the special token and the class token",  
    )
    parser.add_argument(
        "--use_ref_images", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the reference (black bg) images", 
    )
    parser.add_argument(
        "--use_controlnet_images", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the reference (black bg) images", 
    )
    parser.add_argument(
        "--text_encoder_bypass", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to apply the text encoder skip connection",
    )
    parser.add_argument(
        "--appearance_skip_connection", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the appearance skip connection (through the merger mlp)",
    ) 
    parser.add_argument(
        "--replace_attn_maps", 
        type=str, 
        choices=["special2class", "special2class_detached", "class2special", "class2special_detached"], 
        help="whether to replace the special token attention maps by the class token attention maps", 
    ) 
    parser.add_argument(
        "--penalize_special_token_attn", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to penalize the attention maps of the special token against the class token", 
    ) 
    parser.add_argument(
        "--special_token_attn_loss_weight", 
        type=float, 
        required=True, 
        help="the weight of the special token attention loss", 
    ) 
    parser.add_argument(
        "--with_prior_preservation",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        required=True, 
        help="The weight of prior preservation loss.",
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
        "--seed", type=int, default=1510, help="A seed for reproducible training."
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
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_unet",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to train the unet",
    )
    parser.add_argument(
        "--train_text_encoder",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--textual_inv",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to use textual inversion",
    )
    parser.add_argument(
        "--learn_class_embedding",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to learn the class embeddings and use the learnt ones for the reference images?",
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
        required=True, 
        help="Number of steps for stage 1 training", 
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        required=True, 
        help="Number of steps for stage 2 training", 
    )
    parser.add_argument(
        "--merged_emb_dim",
        type=int,
        required=True, 
        help="the output dimension of the merger",  
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
        "--resume_training_state",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    # parser.add_argument(
    #     "--resume_unet",
    #     type=str,
    #     default=None,
    #     help=("File path for unet lora to resume training."),
    # )
    # parser.add_argument(
    #     "--resume_text_encoder",
    #     type=str,
    #     default=None,
    #     help=("File path for text encoder lora to resume training."),
    # )
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
    
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    # else:
    #     if args.class_data_dir is not None:
    #         logger.warning(
    #             "You need not use --class_data_dir without --with_prior_preservation."
    #         )
    #     if args.class_prompt is not None:
    #         logger.warning(
    #             "You need not use --class_prompt without --with_prior_preservation."
    #         )

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

    if osp.exists(args.instance_data_dir_1subject): 
        # subjects_ are the folders in the instance directory 
        subjects_combs_1subject = sorted(os.listdir(args.instance_data_dir_1subject))  
        # args.subjects_combs_1subject = [" ".join(subjects_comb.split("__")) for subjects_comb in subjects_combs_1subject]  
        args.subjects_combs_1subject = subjects_combs_1subject 

    if osp.exists(args.instance_data_dir_2subjects): 
        subjects_combs_2subjects = sorted(os.listdir(args.instance_data_dir_2subjects))  
        # args.subjects_combs_2subjects = [" ".join(subjects_comb.split("__")) for subjects_comb in subjects_combs_2subjects]  
        args.subjects_combs_2subjects = subjects_combs_2subjects  

    # defining the output directory to store checkpoints 
    args.output_dir = osp.join(args.output_dir, f"__{args.run_name}") 

    # storing the number of reference images per subject 
    args.n_ref_imgs = {} 
    if osp.exists(args.instance_data_dir_1subject): 
        for subject_comb_ in args.subjects_combs_1subject: 
            img_files = os.listdir(osp.join(args.instance_data_dir_1subject, subject_comb_)) 
            img_files = [img_file for img_file in img_files if img_file.find("jpg") != -1] 
            args.n_ref_imgs[subject_comb_] = len(img_files)  

    if osp.exists(args.instance_data_dir_2subjects): 
        for subject_comb_ in args.subjects_combs_2subjects: 
            img_files = os.listdir(osp.join(args.instance_data_dir_2subjects, subject_comb_)) 
            img_files = [img_file for img_file in img_files if img_file.find("jpg") != -1] 
            args.n_ref_imgs[subject_comb_] = len(img_files)  

    assert args.merged_emb_dim % 1024 == 0 

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
    # + 1 is added because stage1_steps is set to -1 
    args.max_train_steps = args.stage1_steps + args.stage2_steps + 1  

    # accelerator 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], 
    )

    # effective batch size should remain constant 
    assert accelerator.num_processes * args.train_batch_size == BS, f"{accelerator.num_processes = }, {args.train_batch_size = }" 


    if args.resume_training_state is not None: 
        assert osp.exists(args.resume_training_state) 
        training_state_ckpt = torch.load(args.resume_training_state) 

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

    accelerator.wait_for_everyone() 

    if args.resume_training_state is not None: 
        assert osp.exists(args.resume_training_state) 
        training_state_ckpt = torch.load(args.resume_training_state) 

    if accelerator.is_main_process: 
        pkl_path = osp.join(args.output_dir, f"args.pkl") 
        if osp.exists(pkl_path) and not DEBUG: 
            raise FileExistsError(f"{pkl_path} exists, please delete it first!") 
        with open(pkl_path, "wb") as f: 
            pickle.dump(args.__dict__, f) 

    SAVE_STEPS = [500, 1000, 5000]  
    for save_step in range(SAVE_STEPS_GAP, args.max_train_steps + 1, SAVE_STEPS_GAP): 
        SAVE_STEPS.append(save_step) 
    SAVE_STEPS = sorted(SAVE_STEPS) 

    VLOG_STEPS = [] 
    for vlog_step in range(VLOG_STEPS_GAP, args.max_train_steps + 1, VLOG_STEPS_GAP): 
        VLOG_STEPS.append(vlog_step)
    VLOG_STEPS = sorted(VLOG_STEPS)  

    print(f"{SAVE_STEPS = }") 
    print(f"{VLOG_STEPS = }") 

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
            unet, r=args.lora_rank  
        )
        
        # sanity checks 
        n_lora_params_state_dict = 0 
        for name, param in unet.state_dict().items(): 
            if name.find("lora") != -1: 
                n_lora_params_state_dict += 1 

        assert n_lora_params_state_dict == len(unet_lora_params) 
        # print(f"{n_lora_params_state_dict = }")
        # print(f"{len(unet_lora_params) = }") 
        # sys.exit(0) 

        if args.resume_training_state: 
            # with torch.no_grad(): 
            unet_state_dict = unet.state_dict() 
            lora_state_dict = training_state_ckpt["unet"]["lora"] 
            for name, param in unet_state_dict.items(): 
                if name.find("lora") == -1: 
                    assert name not in lora_state_dict.keys() 
                    continue 
                assert name in lora_state_dict.keys() 
                unet_state_dict[name] = lora_state_dict[name]  
            unet.load_state_dict(unet_state_dict) 

    # retval = patch_custom_attention(unet, store_attn=False, across_timesteps=False, store_loss=args.penalize_special_token_attn)  
    # if args.penalize_special_token_attn: 
    #     assert len(retval) == 1 
    #     loss_store = retval[0] 

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

        # sanity checks 
        n_lora_params_state_dict = 0 
        for name, param in text_encoder.state_dict(): 
            if name.find("lora") != -1: 
                n_lora_params_state_dict += 1 
        assert n_lora_params_state_dict == len(text_encoder_lora_params) 

        if args.resume_training_state: 
            text_encoder_state_dict = text_encoder.state_dict() 
            lora_state_dict = training_state_ckpt["text_encoder"]["lora"] 
            for name, param in text_encoder_state_dict.items():  
                if name.find("lora") == -1: 
                    assert name not in lora_state_dict 
                    continue 
                assert name in lora_state_dict  
                text_encoder_state_dict[name] = lora_state_dict[name]  
            text_encoder.load_state_dict(text_encoder_state_dict) 


    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()


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
        if args.resume_training_state: 
            optimizer_unet.load_state_dict(training_state_ckpt["unet"]["optimizer"]) 
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
        if args.resume_training_state: 
            optimizer_text_encoder.load_state_dict(training_state_ckpt["text_encoder"]["optimizer"]) 
        # optimizers.append(optimizer_text_encoder) 
        optimizers["text_encoder"] = optimizer_text_encoder 

    if args.textual_inv or args.learn_class_embedding: 
        # the appearance embeddings 
        bnha_embeds = {} 
        args.subjects = [
            "jeep", 
            "motorbike", 
            "bus", 
            "lion", 
            "elephant", 
            "horse", 
        ] 
        for subject in args.subjects:  
            # initializing using the subject's embedding in the pretrained CLIP text encoder 
            bnha_embeds[subject] = torch.clone(text_encoder.get_input_embeddings().weight[TOKEN2ID[subject]]).detach()  

        # initializing the AppearanceEmbeddings module using the embeddings 
        bnha_embeds = AppearanceEmbeddings(bnha_embeds).to(accelerator.device) 

        if args.resume_training_state: 
            bnha_embeds.load_state_dict(training_state_ckpt["appearance"]["model"]) 

        # an optimizer for the appearance embeddings 
        optimizer_bnha = optimizer_class(
            bnha_embeds.parameters(),  
            lr=args.learning_rate_emb,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        if args.resume_training_state: 
            optimizer_bnha.load_state_dict(training_state_ckpt["appearance"]["optimizer"]) 
        # optimizers.append(optimizer_bnha) 
        optimizers["appearance"] = optimizer_bnha 


    if not args.pose_only_embedding: 
        pos_size = 2
        continuous_word_model = continuous_word_mlp(input_size=pos_size, output_size=1024)
        if args.resume_training_state: 
            continuous_word_model.load_state_dict(training_state_ckpt["contword"]["model"]) 
        optimizer_mlp = optimizer_class(
            continuous_word_model.parameters(),  
            lr=args.learning_rate_mlp,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        if args.resume_training_state: 
            optimizer_mlp.load_state_dict(training_state_ckpt["contword"]["optimizer"]) 
        # optimizers.append(optimizer_mlp)  
        optimizers["contword"] = optimizer_mlp 


        # the merged token formulation 
        merger = MergedEmbedding(args.appearance_skip_connection, pose_dim=1024, appearance_dim=1024, output_dim=args.merged_emb_dim)    
        if not args.learn_pose and args.resume_training_state: 
            merger.load_state_dict(training_state_ckpt["merger"]["model"]) 
        # optimizer_merger = torch.optim.Adam(merger.parameters(), lr=args.learning_rate_merger)  
        optimizer_merger = optimizer_class(
            merger.parameters(),  
            lr=args.learning_rate_merger,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        if not args.learn_pose and args.resume_training_state: 
            optimizer_merger.load_state_dict(training_state_ckpt["merger"]["optimizer"]) 
        # optimizers.append(optimizer_merger) 
        optimizers["merger"] = optimizer_merger 

    else: 
        continuous_word_model = None 
        if args.use_location_conditioning: 
            merger = PoseLocationEmbedding(256, args.merged_emb_dim) 
        else: 
            merger = PoseEmbedding(output_dim=args.merged_emb_dim)  
        # for name, p in merger.named_parameters(): 
        # REMEMBER THAT THERE IS A RANDOM PROJECTION IN THE GAUSSIAN FOURIER FEATURES, AND HENCE THAT IS NOT LEARNABLE 
        #     print(f"{name = }, {p.shape = }, {p.requires_grad = }") 

        if not args.learn_pose and args.resume_training_state: 
            merger.load_state_dict(training_state_ckpt["merger"]["model"], strict=False)  
        # optimizer_merger = torch.optim.Adam(merger.parameters(), lr=args.learning_rate_merger)  
        optimizer_merger = optimizer_class(
            merger.parameters(),  
            lr=args.learning_rate_merger,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        if not args.learn_pose and args.resume_training_state: 
            optimizer_merger.load_state_dict(training_state_ckpt["merger"]["optimizer"]) 
        # optimizers.append(optimizer_merger) 
        optimizers["merger"] = optimizer_merger 


    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # defining the dataset 
    train_dataset_stage1 = DisentangleDataset(
        args=args, 
        tokenizer=tokenizer, 
        ref_imgs_dirs=[args.instance_data_dir_1subject], 
        controlnet_imgs_dirs=[args.controlnet_data_dir_1subject], 
        num_steps=args.stage1_steps, 
    ) 

    train_dataset_stage2 = DisentangleDataset(
        args=args, 
        tokenizer=tokenizer, 
        ref_imgs_dirs=[args.instance_data_dir_1subject, args.instance_data_dir_2subjects],  
        controlnet_imgs_dirs=[args.controlnet_data_dir_1subject, args.controlnet_data_dir_2subjects],  
        num_steps=args.stage2_steps, 
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
        subjects = [example["subjects"] for example in examples] 
        bboxes = [example["bboxes"] for example in examples] 
        xs_2d = [example["2d_xs"] for example in examples] 
        ys_2d = [example["2d_ys"] for example in examples] 
        pixel_values = []
        for example in examples:
            pixel_values.append(example["img"])

        """Adding the scaler of the embedding into the batch"""
        scalers = [example["scalers"] for example in examples] 

        if args.with_prior_preservation:
            prompt_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_img"] for example in examples]
            prompts += [example["class_prompt"] for example in examples] 
            prior_subjects = [example["prior_subject"] for example in examples] 

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() 

        prompt_ids = tokenizer.pad(
            {"input_ids": prompt_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "prompt_ids": prompt_ids, 
            "pixel_values": pixel_values,
            "scalers": scalers,
            "subjects": subjects, 
            "controlnet": is_controlnet, 
            "prompts": prompts, 
            "2d_xs": xs_2d, 
            "2d_ys": ys_2d, 
            "bboxes": bboxes, 
        }
        if args.with_prior_preservation: 
            batch["prior_subjects"] = prior_subjects  

        return batch 
    """end Adobe CONFIDENTIAL"""

    train_dataloader_stage1 = torch.utils.data.DataLoader(
        train_dataset_stage1,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=accelerator.num_processes * 2,
    )

    train_dataloader_stage2 = torch.utils.data.DataLoader(
        train_dataset_stage2,
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
    
    
    unet, text_encoder, merger, continuous_word_model, train_dataloader_stage1, train_dataloader_stage2 = accelerator.prepare(unet, text_encoder, merger, continuous_word_model, train_dataloader_stage1, train_dataloader_stage2)   
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
    if continuous_word_model is not None: 
        continuous_word_model.to(accelerator.device)  
    """End Adobe CONFIDENTIAL"""

    for name, param in unet.state_dict().items(): 
        if name.find("lora") == -1: 
            param.to(accelerator.device, dtype=weight_dtype) 
    
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
    if args.train_text_encoder: 
        text_encoder.train() 
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
    if continuous_word_model is not None: 
        continuous_word_model.train()

    merger.train() 
    """End Adobe CONFIDENTIAL"""

    # steps_per_angle = {} 
    input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight).detach()  

    # steps_per_angle = {} 
    if DEBUG and accelerator.is_main_process: 
        if osp.exists(f"vis"): 
            shutil.rmtree(f"vis") 
        os.makedirs("vis")  

    train_dataloader_stage1_iter = iter(train_dataloader_stage1) 
    train_dataloader_stage2_iter = iter(train_dataloader_stage2) 


    while True: 

        retval = patch_custom_attention(accelerator.unwrap_model(unet), store_attn=False, across_timesteps=False, store_loss=args.penalize_special_token_attn)  
        loss_store = retval["loss_store"] 
        attn_store = retval["attn_store"] 
        if args.penalize_special_token_attn: 
            assert loss_store is not None and attn_store is None 
            # assert that we are beginning with an empty loss store 
            assert loss_store.step_store["loss"] == 0.0 
        # for batch_idx, angle in enumerate(batch["anagles"]): 
        #     if angle in steps_per_angle.keys(): 
        #         steps_per_angle[angle] += 1 
        #     else:
        #         steps_per_angle[angle] = 1 
        if args.resume_training_state: 
            if global_step < training_state_ckpt["global_step"]:  
                global_step += BS  
                progress_bar.update(BS) 
                ddp_step += 1 
                if accelerator.is_main_process and args.wandb: 
                    wandb_log_data = {} 
                    for _ in range(BS):  
                        wandb.log(wandb_log_data) 
                continue 

        if global_step <= args.stage1_steps:  
            MAX_SUBJECTS_PER_EXAMPLE = 1  
            batch = next(train_dataloader_stage1_iter)  
        else: 
            MAX_SUBJECTS_PER_EXAMPLE = 2   
            batch = next(train_dataloader_stage2_iter)  

        if DEBUG: 
            assert torch.allclose(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight, input_embeddings) 

        # print(f"{batch.keys()}")
        B = len(batch["scalers"])   

        if PRINT_STUFF: 
            accelerator.print(f"<=============================== step {global_step}  ======================================>")
            for key, value in batch.items(): 
                if ("ids" in key) or ("values" in key): 
                    accelerator.print(f"{key}: {value.shape}") 
                else:
                    accelerator.print(f"{key}: {value}") 
            accelerator.print(f"{MAX_SUBJECTS_PER_EXAMPLE = }") 

            # making some checks on the dataloader outputs in case of DEBUG mode 
            # if DEBUG: 
            #     if "ids" in key: 
            #         # this is necessary because we are on a "nosubject" formulation 
            #         for batch_idx in range(B):  
            #             # print(f"{B = }")
            #             # print(f"{batch_idx = }")
            #             # print(f"{value.shape = }") 
            #             assert TOKEN2ID[batch["subjects"][batch_idx]] not in value[batch_idx], f"{batch['subjects'][batch_idx] = }, {batch['prompts'][batch_idx] = }"   
            #             assert TOKEN2ID["bnha"] in value 

        if DEBUG or args.wandb: 
            wandb_log_data = {}
            force_wandb_log = False 
        # Convert images to latent space
        vae.to(accelerator.device, dtype=weight_dtype)  

        if DEBUG and accelerator.is_main_process: 
            for batch_idx, img_t in enumerate(batch["pixel_values"]): 
                img = (img_t * 0.5 + 0.5) * 255  
                img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8) 
                img = np.ascontiguousarray(img) 
                locations = [] 

                bboxes = batch["bboxes"]  
                if batch_idx < len(batch['2d_xs']): 
                    for asset_idx in range(len(batch["subjects"][batch_idx])): 
                        location = (int(batch["2d_xs"][batch_idx][asset_idx] * 512), int(batch["2d_ys"][batch_idx][asset_idx] * 512))   
                        locations.append(location) 
                        if PRINT_STUFF: 
                            accelerator.print(f"drawing circle at {location}!") 
                        bbox = bboxes[batch_idx][asset_idx]
                        tl = (int(bbox[0] * 512), int(bbox[1] * 512))  
                        br = (int(bbox[2] * 512), int(bbox[3] * 512))  
                        cv2.circle(img, location, 0, (255, 0, 0), 10) 
                        cv2.rectangle(img, tl, br, (0, 255, 0), 5) 

                plt.figure(figsize=(20, 20)) 
                plt.imshow(img)  
                if batch_idx < B: 
                    plt_title = f"{global_step = }\t{batch_idx = }\t{batch['prompts'][batch_idx] = }\t{batch['subjects'][batch_idx] = }\t{batch['scalers'][batch_idx] = }" 
                else: 
                    plt_title = f"{global_step = }\t{batch_idx = }\t{batch['prompts'][batch_idx] = }" 
                plt_title = "\n".join(textwrap.wrap(plt_title, width=60)) 
                plt.title(plt_title, fontsize=9)  
                plt.savefig(f"vis/{str(global_step).zfill(3)}_{str(batch_idx).zfill(3)}.jpg") 
                plt.close() 

        latents = vae.encode(
            batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)  
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
        # if global_step > args.stage1_steps: 
        # we are no longer learning appearance embeddings first! 
        if True: 
            # progress_bar.set_description(f"stage 2: ")
            scalers_padded = torch.zeros((len(batch["scalers"]), MAX_SUBJECTS_PER_EXAMPLE))  
            xs_2d_padded = torch.zeros((len(batch["2d_xs"]), MAX_SUBJECTS_PER_EXAMPLE)) 
            ys_2d_padded = torch.zeros((len(batch["2d_ys"]), MAX_SUBJECTS_PER_EXAMPLE)) 

            for batch_idx in range(len(batch["scalers"])): 
                for scaler_idx in range(len(batch["scalers"][batch_idx])): 
                    scalers_padded[batch_idx][scaler_idx] = batch["scalers"][batch_idx][scaler_idx] 

            assert len(batch["scalers"]) == len(batch["2d_xs"]) == len(batch["2d_ys"])  
            for batch_idx in range(len(batch["scalers"])): 
                assert len(batch["scalers"][batch_idx]) == len(batch["2d_xs"][batch_idx]) == len(batch["2d_ys"][batch_idx])  
                for scaler_idx in range(len(batch["scalers"][batch_idx])): 
                    xs_2d_padded[batch_idx][scaler_idx] = batch["2d_xs"][batch_idx][scaler_idx]  
                    ys_2d_padded[batch_idx][scaler_idx] = batch["2d_ys"][batch_idx][scaler_idx] 

            p = torch.Tensor(scalers_padded / (2 * math.pi)) 
            assert torch.all(xs_2d_padded < 1) 
            assert torch.all(ys_2d_padded < 1) 
            if not args.pose_only_embedding: 
                p = p.unsqueeze(-1) 
                p = p.repeat(1, 1, 2)  
                p[..., 0] = torch.sin(2 * torch.pi * p[..., 0]) 
                p[..., 1] = torch.cos(2 * torch.pi * p[..., 1]) 
                mlp_emb = continuous_word_model(p) 
            else: 
                # mlp_emb = torch.zeros((B, MAX_SUBJECTS_PER_EXAMPLE, args.merged_emb_dim)) 
                mlp_emb = [] 
                if PRINT_STUFF: 
                    accelerator.print(f"{p = }") 
                    accelerator.print(f"{xs_2d_padded = }") 
                    accelerator.print(f"{ys_2d_padded = }") 
                for scaler_idx in range(MAX_SUBJECTS_PER_EXAMPLE):  
                    # mlp_emb[:, scaler_idx, :] = merger(p[:, scaler_idx]) 
                    if args.use_location_conditioning: 
                        mlp_emb.append(merger(p[:, scaler_idx], xs_2d_padded[:, scaler_idx], ys_2d_padded[:, scaler_idx]).unsqueeze(1)) 
                    else: 
                        mlp_emb.append(merger(p[:, scaler_idx]).unsqueeze(1)) 

                mlp_emb = torch.cat(mlp_emb, dim=1) 
                assert mlp_emb.shape == (B, MAX_SUBJECTS_PER_EXAMPLE, args.merged_emb_dim) 
                    

            # getting the embeddings from the mlp

        else: 
            progress_bar.set_description(f"stage 1: ")
            # mlp_emb = torch.zeros(B, 1024) 
            mlp_emb = torch.zeros((B, MAX_SUBJECTS_PER_EXAMPLE, 1024)).to(accelerator.device)  

        num_assets_in_batch = 0 
        for batch_idx in range(B): 
            num_assets_in_batch = num_assets_in_batch + len(batch["scalers"][batch_idx]) 

        # appearance embeddings
        # textual inversion is used, then the embeddings are initialized with their classes  
        # else it is initialized with the default value for bnha 
        if not args.pose_only_embedding: 
            bnha_emb = torch.zeros((len(batch["subjects"]), MAX_SUBJECTS_PER_EXAMPLE, 1024)) 
            if args.textual_inv: 
                # assert False 
                # bnha_emb = torch.stack([getattr(accelerator.unwrap_model(bnha_embeds), subject) for subject in batch["subjects"]])  
                # bnha_emb = torch.stack([bnha_embeds(subject) for subject in batch["subjects"]])  
                assert len(batch["controlnet"]) == B 
                for batch_idx in range(B): 
                    if batch["controlnet"][batch_idx]: 
                        # if controlnet image, then replace the appearance embedding by the class embedding
                        for asset_idx, subject in enumerate(batch["subjects"][batch_idx]): 
                            bnha_emb[batch_idx][asset_idx] = torch.clone(input_embeddings)[TOKEN2ID[subject]]  
                    else: 
                        # bnha_emb.append(bnha_embeds(batch["subjects"][idx])) 
                        for asset_idx, subject in enumerate(batch["subjects"][batch_idx]): 
                            bnha_emb[batch_idx][asset_idx] = getattr(accelerator.unwrap_model(bnha_embeds), subject)  
                # bnha_emb = torch.stack(bnha_emb) 

            else: 
                # bnha_emb = torch.clone(input_embeddings).detach()[TOKEN2ID["bnha"]].unsqueeze(0).repeat(B, 1)  
                # bnha_emb = torch.clone(input_embeddings)[TOKEN2ID[]] 
                for batch_idx in range(B): 
                    # REPLACING BY THE CLASS EMBEDDING 
                    # bnha_emb.append(torch.clone(input_embeddings)[TOKEN2ID[batch["subjects"][idx]]].detach())  
                    for asset_idx, subject in enumerate(batch["subjects"][batch_idx]): 
                        bnha_emb[batch_idx][asset_idx] = torch.clone(input_embeddings)[TOKEN2ID[subject]]  
                # bnha_emb = torch.stack(bnha_emb) 

            merged_emb = merger(mlp_emb, bnha_emb)  
            merged_emb_norm = torch.linalg.norm(merged_emb)  
            assert merged_emb.shape[0] == B 

        else: 
            merged_emb = mlp_emb 
            merged_emb_norm = torch.linalg.norm(merged_emb)   
            assert merged_emb.shape[0] == B 

        # print(f"{bnha_emb.shape = }")
        # print(f"{mlp_emb.shape = }")
        # merging the appearance and pose embeddings 
        # normalizing the merged embeddings 
        # with torch.no_grad(): 
        #     if args.normalize_merged_embedding: 
        #         for batch_idx in range(B): 
        #             for asset_idx in range(len(batch["subjects"][batch_idx])): 
        #                 subject = batch["subjects"][batch_idx][asset_idx] 
        #                 merged_vec = merged_emb[batch_idx][asset_idx]  
        #                 for token_idx in range(args.merged_emb_dim // 1024):  
        #                     merged_vec_slice = merged_vec[token_idx * 1024 : (token_idx+1) * 1024]  
        #                     merged_vec_slice_norm = torch.linalg.norm(merged_vec_slice)  
        #                     org_emb_norm = torch.linalg.norm(input_embeddings[TOKEN2ID[subject]]) 
        #                     # merged_vec_slice = merged_vec_slice * org_emb_norm / merged_vec_slice_norm 
        #                     merged_vec_slice = merged_vec_slice * 2.0  
        #                     # replacing back!
        #                     merged_emb[batch_idx][asset_idx][token_idx * 1024 : (token_idx+1) * 1024] = merged_vec_slice 


        # pose_emb_norm = torch.linalg.norm(mlp_emb) * num_assets_in_batch / (MAX_SUBJECTS_PER_EXAMPLE * B)   
        pose_emb_norm = torch.linalg.norm(mlp_emb)  

        # replacing the input embedding for sks by the mlp for each batch item, and then getting the output embeddings of the text encoder 
        # must run a for loop here, first changing the input embeddings of the text encoder for each 
        encoder_hidden_states = [] 
        attn_assignments = [] 
        if args.with_prior_preservation: 
            input_ids, input_ids_prior = torch.chunk(batch["prompt_ids"], 2, dim=0) 
        else: 
            input_ids = batch["prompt_ids"] 

        for batch_idx, batch_item in enumerate(input_ids): 
            # replacing the text encoder input embeddings by the original ones and setting them to be COLD -- to enable replacement by a hot embedding  
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(torch.clone(input_embeddings), requires_grad=False)  

            # performing the replacement on cold embeddings by a hot embedding -- allowed 
            example_merged_emb = merged_emb[batch_idx] 
            for asset_idx, subject in enumerate(batch["subjects"][batch_idx]):   

                if args.learn_class_embedding: 
                    if batch["controlnet"][batch_idx] == True: 
                        continue 
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[batch["subjects"][batch_idx][asset_idx]]] = getattr(accelerator.unwrap_model(bnha_embeds), batch["subjects"][batch_idx][asset_idx]) 
                        

                for token_idx in range(args.merged_emb_dim // 1024):  
                    # replacement_emb = torch.clone(merged_emb[batch_idx][asset_idx][token_idx * 1024 : (token_idx+1) * 1024])  
                    if args.normalize_merged_embedding: 
                        replacement_mask = torch.ones_like(example_merged_emb, requires_grad=False)      
                        replacement_emb_norm = torch.linalg.norm(example_merged_emb[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]).detach()   
                        org_emb_norm = torch.linalg.norm(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]]).detach()  
                        replacement_mask[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] = org_emb_norm / replacement_emb_norm  
                        assert example_merged_emb.shape == replacement_mask.shape  
                        assert torch.allclose(torch.linalg.norm((example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]), org_emb_norm, atol=1e-3), f"{torch.linalg.norm((example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]) = }, {org_emb_norm = }" 
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = (example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] 
                    else: 
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = (example_merged_emb)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] 

            text_embeddings = text_encoder(batch_item.unsqueeze(0))[0].squeeze() 

            attn_assignments_batchitem = {} 
            if args.learn_pose: 
                unique_token_positions = {}  
                for asset_idx in range(len(batch["subjects"][batch_idx])):  
                    for token_idx in range(args.merged_emb_dim // 1024): 
                        unique_token = UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"] 
                        assert TOKEN2ID[unique_token] in list(batch_item), f"{unique_token = }" 
                        unique_token_idx = list(batch_item).index(TOKEN2ID[unique_token]) 
                        attn_assignments_batchitem[unique_token_idx] = unique_token_idx + args.merged_emb_dim // 1024 - token_idx 
                        unique_token_positions[f"{asset_idx}_{token_idx}"] = unique_token_idx  
            else: 
                for asset_idx in range(len(batch["subjects"][batch_idx])):  
                    subject = batch["subjects"][batch_idx][asset_idx] 
                    subject_token_idx = list(batch_item).index(TOKEN2ID[subject]) 
                    attn_assignments_batchitem[subject_token_idx] = subject_token_idx 

            attn_assignments.append(attn_assignments_batchitem) 

            if args.text_encoder_bypass: 
                for unique_token_name, position in unique_token_positions.items(): 
                    text_embeddings[position] = text_embeddings[position] + accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[unique_token_name]]]  

            encoder_hidden_states.append(text_embeddings)  

        encoder_hidden_states = torch.stack(encoder_hidden_states)  

        # replacing the text encoder input embeddings by the original ones, this time setting them to be HOT, this will be useful in case we choose to do textual inversion 
        # here we are not cloning because these won't be stepped upon anyways, and this way we can save some memory also!  
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(torch.clone(input_embeddings), requires_grad=False)   
        if args.with_prior_preservation: 
            encoder_hidden_states_prior = text_encoder(input_ids_prior)[0] 
            assert encoder_hidden_states_prior.shape == encoder_hidden_states.shape 
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_prior], dim=0) 
            assert len(input_ids_prior) == args.train_batch_size, f"{len(input_ids_prior) = }, {args.train_batch_size = }" 
            for _ in range(args.train_batch_size):  
                attn_assignments.append({}) 

        """End Adobe CONFIDENTIAL"""


        # Predict the noise residual
        if args.ada: 
            torch.cuda.empty_cache() 

        encoder_states_dict = {
            "encoder_hidden_states": encoder_hidden_states, 
            "attn_assignments": attn_assignments, 
        } 
        if args.replace_attn_maps is not None: 
            encoder_states_dict[args.replace_attn_maps] = True 

        if args.penalize_special_token_attn: 
            encoder_states_dict["bboxes"] = batch["bboxes"] 

        if args.attn_bbox_from_class_mean: 
            encoder_states_dict["bbox_from_class_mean"] = True 
            encoder_states_dict["bboxes"] = batch["bboxes"] 

        # if args.replace_attn_maps is not None or args.penalize_special_token_attn or args.bbox_from_class_mean:  
        if DEBUG: 
            os.makedirs(osp.join("vis_attnmaps", f"{str(global_step).zfill(3)}"), exist_ok=True) 
        model_pred = unet(noisy_latents, timesteps, encoder_states_dict).sample 
        # else: 
        #     model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.penalize_special_token_attn: 
            assert loss_store.step_store["loss"].device == accelerator.device 
            loss_store.step_store["loss"] = loss_store.step_store["loss"] / args.train_batch_size 

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
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.append(loss.detach()) 

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            losses.append(prior_loss.detach() * args.prior_loss_weight) 

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.append(loss.detach()) 
            losses.append(torch.tensor(0.0).to(accelerator.device)) 

        if args.penalize_special_token_attn: 
            losses.append(loss_store.step_store["loss"].detach() * args.special_token_attn_loss_weight)  
            loss = loss + args.special_token_attn_loss_weight * loss_store.step_store["loss"] 
        else: 
            losses.append(torch.tensor(0.0).to(accelerator.device))   

        if PRINT_STUFF: 
            accelerator.print(f"MSE loss: {losses[0].item()}, the weight is 1.0")
            accelerator.print(f"prior loss: {losses[1].item()}, {args.prior_loss_weight = }") 
            accelerator.print(f"special token attn loss: {losses[2].item()}, {args.special_token_attn_loss_weight = }") 


        losses = torch.stack(losses).to(accelerator.device) 

        if args.ada: 
            torch.cuda.empty_cache() 

        # checking if the parameters do require grads at least 
        # if DEBUG: 
        #     for p in merger.parameters(): 
        #         assert p.requires_grad 


        accelerator.backward(loss)
        # everytime the continuous word mlp must receive gradients 
        if DEBUG: 
            with torch.no_grad(): 
                # checking that merger receives gradients 
                bad_merger_params = [(n, p) for (n, p) in merger.named_parameters() if p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))] 
                bad_merger_params = [(n, p) for (n, p) in bad_merger_params if n.find("gaussian_fourier_embedding") == -1] 
                # assert len(bad_merger_params) == 0, f"{len(bad_merger_params) = }, {len(list(merger.parameters())) = }" 
                # print(f"{len(bad_merger_params) = }, {len(list(merger.parameters())) = }")  
                # for (n, p) in merger.named_parameters():  
                    # if (n, p) not in bad_merger_params:  
                        # print(f"{n, p = } in merger is NOT bad!")
                # if global_step < args.stage1_steps: 
                #     assert len(bad_merger_params) < len(list(merger.parameters()))  
                if global_step > 1 and args.learn_pose: 
                    assert len(bad_merger_params) == 0, f"{len(bad_merger_params) = }" 
                    # print(f"{len(bad_merger_params) = }") 

                # checking that mlp receives gradients in stage 2 
                # print(f"merger does receive gradients!")
                if not args.pose_only_embedding and args.learn_pose: 
                    bad_mlp_params = [(n, p) for (n, p) in continuous_word_model.named_parameters() if p.grad is None or torch.allclose(p.grad, torch.tensor(0.0).to(accelerator.device))]   
                    # assert not ((len(bad_mlp_params) < len(list(continuous_word_model.parameters()))) ^ (global_step > args.stage1_steps))  
                    # assert not ((len(bad_mlp_params) == 0) ^ (global_step > args.stage1_steps))  
                    if global_step > args.stage1_steps + 2:  
                        # print(f"{len(bad_mlp_params) = }, {len(list(continuous_word_model.parameters())) = }")  
                        # assert len(bad_mlp_params) < len(list(continuous_word_model.parameters()))  
                        assert len(bad_mlp_params) == 0  
                # print(f"{len(bad_mlp_params) = }") 
                    # print(f"mlp does receive gradients!")

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
                    

                # checking whether the text encoder will receive gradients 
                # if args.train_text_encoder: 
                #     # some_grad_is_good = False  
                #     # for p in list(text_encoder.parameters()):   
                #     for p in text_encoder_lora_params:   
                #         if p.grad is None: 
                #             continue 
                #         # if not torch.allclose(p.grad, torch.zeros_like(p.grad)):   
                #         #     some_grad_is_good = True 
                #         assert not torch.allclose(p.grad, torch.zeros_like(p.grad))  
                #     # assert some_grad_is_good 

                # # checking whether the unet will receive gradients 
                # if args.train_unet: 
                #     # some_grad_is_good = False 
                #     # for p in list(itertools.chain(*unet_lora_params)):    
                #     for n, p in list(unet.named_parameters()):    
                #         if p.grad is None: 
                #             continue 
                #         # print(f"{torch.zeros_like(p.grad) = }, {p.grad = }")
                #         # print(f"something is not none also!")
                #         if not torch.allclose(p.grad, torch.zeros_like(p.grad)):  
                #             # print(f"{n = } has a gradient!")
                #             some_grad_is_good = True 
                #         else: 
                #             # assert not torch.allclose(p.grad, torch.zeros_like(p.grad)) 
                #             # print(f"{n = } DOES NOT HAVE GRADIENT...")
                #             pass 
                #     assert some_grad_is_good 


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

                if not args.pose_only_embedding: 
                    # mlp 
                    mlp_grad_norm = [torch.linalg.norm(param.grad) for param in continuous_word_model.parameters() if param.grad is not None]
                    if len(mlp_grad_norm) == 0:
                        mlp_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        mlp_grad_norm = torch.mean(torch.stack(mlp_grad_norm)) 
                    all_grad_norms.append(mlp_grad_norm) 
                else: 
                    all_grad_norms.append(torch.tensor(-1.0).to(accelerator.device))   


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
                else: 
                    all_grad_norms.append(torch.tensor(0.0).to(accelerator.device))
                        
                # text encoder 
                if args.train_text_encoder: 
                    text_encoder_grad_norm = [torch.linalg.norm(param.grad) for param in text_encoder.parameters() if param.grad is not None]
                    if len(text_encoder_grad_norm) == 0:
                        text_encoder_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        text_encoder_grad_norm = torch.mean(torch.stack(text_encoder_grad_norm)) 
                    all_grad_norms.append(text_encoder_grad_norm) 
                else: 
                    all_grad_norms.append(torch.tensor(0.0).to(accelerator.device))
                
                # embedding  
                if args.textual_inv: 
                    bnha_grad_norm = [torch.linalg.norm(param.grad) for param in bnha_embeds.parameters() if param.grad is not None] 
                    if len(bnha_grad_norm) == 0: 
                        bnha_grad_norm = torch.tensor(0.0).to(accelerator.device) 
                    else: 
                        bnha_grad_norm = torch.mean(torch.stack(bnha_grad_norm)) 
                    all_grad_norms.append(bnha_grad_norm) 
                else: 
                    all_grad_norms.append(torch.tensor(0.0).to(accelerator.device))


                # grad_norms would be in the order (if available): mlp, unet, text_encoder, embedding  
                # gathering all the norms at once to prevent excessive multi gpu communication 
                all_grad_norms = torch.stack(all_grad_norms).unsqueeze(0)  
                gathered_grad_norms = torch.mean(accelerator.gather(all_grad_norms), dim=0)  
                wandb_log_data["mlp_grad_norm"] = gathered_grad_norms[0] 
                wandb_log_data["merger_grad_norm"] = gathered_grad_norms[1]  
                curr = 2  
                while curr < len(gathered_grad_norms):  
                    # if args.train_unet and ("unet_grad_norm" not in wandb_log_data.keys()): 
                    if ("unet_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["unet_grad_norm"] = gathered_grad_norms[curr]  

                    elif ("text_encoder_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["text_encoder_grad_norm"] = gathered_grad_norms[curr] 

                    elif ("bnha_grad_norm" not in wandb_log_data.keys()): 
                        wandb_log_data["bnha_grad_norm"] = gathered_grad_norms[curr] 
                    
                    else:
                        assert False 
                    curr += 1

        # gradient clipping 
        # if accelerator.sync_gradients:
        #     params_to_clip = [] 
        #     parmas_to_clip = params_to_clip + list(itertools.chain(continuous_word_model.parameters())) + list(itertools.chain(merger.parameters()))  
        #     if args.train_unet: 
        #         params_to_clip = parmas_to_clip + list(itertools.chain(unet.parameters()))  
        #     if args.train_text_encoder: 
        #         params_to_clip = parmas_to_clip + list(itertools.chain(text_encoder.parameters()))  
        #     if args.textual_inv: 
        #         params_to_clip = params_to_clip + list(itertools.chain(bnha_embeds.parameters())) 
        #     # params_to_clip = (
        #     #     itertools.chain(unet.parameters(), text_encoder.parameters(), continuous_word_model.parameters())
        #     #     if args.train_text_encoder
        #     #     else itertools.chain(unet.parameters(), continuous_word_model.parameters())
        #     # )
        #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        if DEBUG: 
            with torch.no_grad(): 
                merger_before = copy.deepcopy([p for p in merger.parameters()]) 
                if not args.pose_only_embedding: 
                    mlp_before = copy.deepcopy([p for p in continuous_word_model.parameters()])  
                unet_before = copy.deepcopy([p for p in unet.parameters()]) 
                text_encoder_before = copy.deepcopy([p for p in text_encoder.parameters()]) 
                if args.textual_inv: 
                    bnha_before = copy.deepcopy([p for p in bnha_embeds.parameters()]) 

        # lora_before = [torch.clone(p) for p in list(itertools.chain(*unet_lora_params))] 
        for name, optimizer in optimizers.items(): 
            optimizer.step() 

        # calculating weight norms 
        if args.wandb and ((ddp_step + 1) % args.log_every == 0): 
            with torch.no_grad(): 
                all_norms = []

                # mlp 
                if not args.pose_only_embedding: 
                    mlp_norm = [torch.linalg.norm(param) for param in continuous_word_model.parameters() if param.grad is not None]
                    if len(mlp_norm) == 0:
                        mlp_norm = torch.tensor(0.0).to(accelerator.device) 
                    else:
                        mlp_norm = torch.mean(torch.stack(mlp_norm)) 
                    all_norms.append(mlp_norm) 
                else: 
                    all_norms.append(torch.tensor(-1.0).to(accelerator.device))  


                # merger  
                merger_norm = [torch.linalg.norm(param) for param in merger.parameters() if param.grad is not None]
                if len(merger_norm) == 0:
                    merger_norm = torch.tensor(0.0).to(accelerator.device) 
                else:
                    merger_norm = torch.mean(torch.stack(merger_norm)) 
                all_norms.append(merger_norm) 

                # merged_embedding norm 
                all_norms.append(merged_emb_norm)  

                all_norms.append(pose_emb_norm) 

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
                wandb_log_data["pose_emb_norm"] = gathered_norms[3] 
                curr = 4   
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

        if DEBUG: 
            # checking that no parameter should be NaN 
            for p in merger.parameters(): 
                assert not torch.any(torch.isnan(p)), f"{[n for (n, p) in merger.named_parameters() if torch.any(torch.isnan(p))]}" 
            if not args.pose_only_embedding: 
                for p in continuous_word_model.parameters(): 
                    assert not torch.any(torch.isnan(p)) 
            for p in unet.parameters(): 
                assert not torch.any(torch.isnan(p)) 
            for p in text_encoder.parameters(): 
                assert not torch.any(torch.isnan(p)) 

            with torch.no_grad(): 
                merger_after = [p for p in merger.parameters()]  
                if not args.pose_only_embedding: 
                    mlp_after = [p for p in continuous_word_model.parameters()]  
                unet_after = [p for p in unet.parameters()]  
                text_encoder_after = [p for p in text_encoder.parameters()]  
                if args.textual_inv: 
                    bnha_after = [p for p in bnha_embeds.parameters()]  

                merger_after = [p1 - p2 for p1, p2 in zip(merger_before, merger_after)] 
                del merger_before 
                if not args.pose_only_embedding: 
                    mlp_after = [p1 - p2 for p1, p2 in zip(mlp_before, mlp_after)] 
                    del mlp_before 
                unet_after = [p1 - p2 for p1, p2 in zip(unet_before, unet_after)]  
                del unet_before 
                text_encoder_after = [p1 - p2 for p1, p2 in zip(text_encoder_before, text_encoder_after)]  
                del text_encoder_before
                if args.textual_inv: 
                    bnha_after = [p1 - p2 for p1, p2 in zip(bnha_before, bnha_after)]   
                    del bnha_before 

                change = False 
                for p_diff in merger_after: 
                    if not torch.allclose(p_diff, torch.zeros_like(p_diff)):  
                        change = True 
                        break 
                assert change 

                # change = False 
                # for p_diff in mlp_after:  
                #     if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
                #         change = True 
                #         break 
                # assert not (change ^ (global_step > args.stage1_steps)), f"{change = }, {global_step = }, {args.stage1_steps = }" 

                change = False 
                for p_diff in unet_after:  
                    if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
                        change = True 
                        break 
                assert not (change ^ args.train_unet)  

                change = False 
                for p_diff in text_encoder_after:  
                    if not torch.allclose(p_diff, torch.zeros_like(p_diff)):   
                        change = True 
                        break 
                assert not (change ^ args.train_text_encoder)   
            
                if args.textual_inv and torch.sum(torch.tensor(batch["controlnet"])).item() < B:  
                    change = False 
                    for p_diff in bnha_after:  
                        if not torch.allclose(p_diff, torch.zeros_like(p_diff)):  
                            change = True 
                            break 
                    assert not (change ^ args.textual_inv), f"{batch['controlnet'] = }" 


        progress_bar.update(accelerator.num_processes * args.train_batch_size) 

        # optimizer_unet.zero_grad()
        # optimizer_text_encoder.zero_grad()
        # continuous_word_optimizer.zero_grad()
        for name, optimizer in optimizers.items(): 
            optimizer.zero_grad() 

        """end Adobe CONFIDENTIAL"""

        # since we have stepped, time to log weight norms!

        global_step += accelerator.num_processes * args.train_batch_size  
        # ddp_step += 1 

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
                if not args.pose_only_embedding: 
                    mlp_params_safe = [torch.clone(p) for p in continuous_word_model.parameters()] 
                merger_params_safe = [torch.clone(p) for p in merger.parameters()] 
                # bnha_embeds_safe = [torch.clone(p) for p in bnha_embeds.parameters()] 


            # if (DEBUG or args.wandb) and args.textual_inv and args.online_inference: 
            #     wandb_log_data = infer(args, step, wandb_log_data, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, merger, bnha_embeds, input_embeddings) 
            #     force_wandb_log = True 
            #     set_seed(args.seed + accelerator.process_index) 
            # elif (DEBUG or args.wandb) and args.online_inference: 
            # ONLY PERFORMING CLASS INFERENCE HERE!  
            # print(f"just before infer function call, {MAX_SUBJECTS_PER_EXAMPLE = }")
            wandb_log_data = infer(args, step, wandb_log_data, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, merger, None, input_embeddings, MAX_SUBJECTS_PER_EXAMPLE) 
            force_wandb_log = True 
            set_seed(args.seed + accelerator.process_index) 
            torch.cuda.empty_cache() 

            if DEBUG: 
                for p_, p in zip(unet_params_safe, unet.parameters()): 
                    assert torch.allclose(p_, p) 
                for p_, p in zip(text_encoder_params_safe, text_encoder.parameters()):  
                    assert torch.allclose(p_, p) 
                if not args.pose_only_embedding: 
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

                    training_state["global_step"] = global_step 

                    training_state["appearance"] = {} 
                    if not args.pose_only_embedding: 
                        training_state["contword"] = {} 
                    training_state["merger"] = {} 
                    training_state["text_encoder"] = {} 
                    training_state["unet"] = {} 

                    if args.train_unet: 
                        unet_lora_state_dict = {} 
                        for name, param in accelerator.unwrap_model(unet).state_dict().items(): 
                            if name.find(f"lora") == -1: 
                                continue 
                            unet_lora_state_dict[name] = param 

                    if args.train_text_encoder: 
                        text_encoder_lora_state_dict = {} 
                        for name, param in accelerator.unwrap_model(text_encoder).state_dict().items(): 
                            if name.find(f"lora") == -1: 
                                continue 
                            text_encoder_lora_state_dict[name] = param 

                    training_state["merger"]["optimizer"] = optimizers["merger"].state_dict() 
                    training_state["merger"]["model"] = accelerator.unwrap_model(merger).state_dict() 

                    if not args.pose_only_embedding: 
                        training_state["contword"]["optimizer"] = optimizers["contword"].state_dict() 
                        training_state["contword"]["model"] = accelerator.unwrap_model(continuous_word_model).state_dict() 

                    if args.textual_inv: 
                        training_state["appearance"]["optimizer"] = optimizers["appearance"].state_dict() 
                        training_state["appearance"]["model"] = accelerator.unwrap_model(bnha_embeds).state_dict()  

                    if args.train_unet: 
                        training_state["unet"]["optimizer"] = optimizers["unet"].state_dict() 
                        training_state["unet"]["model"] = args.pretrained_model_name_or_path  
                        # training_state["unet"]["lora"] = list(itertools.chain(*unet_lora_params)) 
                        training_state["unet"]["lora"] = unet_lora_state_dict  

                    if args.train_text_encoder: 
                        training_state["text_encoder"]["optimizer"] = optimizers["text_encoder"].state_dict() 
                        training_state["text_encoder"]["model"] = args.pretrained_model_name_or_path  
                        # training_state["text_encoder"]["lora"] = list(itertools.chain(*text_encoder_lora_params)) 
                        training_state["text_encoder"]["lora"] = text_encoder_lora_state_dict  

                    save_dir = osp.join(args.output_dir, f"training_state_{global_step}.pth")
                    torch.save(training_state, save_dir)   

                    accelerator.print(f"<=========== SAVED CHECKPOINT FOR STEP {global_step} ===============>") 

                    # this is for saving the safeloras 
                    # loras = {}
                    # if args.train_unet: 
                    #     loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})

                    # print("Cross Attention is also updated!")

                    # # """ If updating only cross attention """
                    # # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})

                    # if args.train_text_encoder:
                    #     loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

                    # if loras != {}: 
                    #     save_safeloras(loras, f"{args.output_dir}/lora_weight_{global_step}.safetensors")
                    
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
        if args.wandb and (ddp_step + 1) % args.log_every == 0: 
            # wandb_log_data["loss"] = gathered_loss
            wandb_log_data["scaled_mse_loss"] = gathered_losses[0]   
            wandb_log_data["scaled_prior_loss"] = gathered_losses[1] 
            wandb_log_data["scaled_special_token_attn_loss"] = gathered_losses[2] 

        if args.wandb: 
            # finally logging!
            if accelerator.is_main_process and (force_wandb_log or (ddp_step + 1) % args.log_every == 0): 
                for logging_step in range(global_step - BS, global_step - 1): 
                    wandb.log({
                        "global_step": logging_step, 
                    })
                wandb_log_data["global_step"] = global_step 
                wandb.log(wandb_log_data) 

            # hack to make sure that the wandb step and the global_step are in sync 
            elif accelerator.is_main_process: 
                for logging_step in range(global_step - BS, global_step): 
                    wandb.log({
                        "global_step": logging_step, 
                    })

        ddp_step += 1 

        logs = {"loss": gathered_loss.item()} 

        progress_bar.set_postfix(**logs)
        # accelerator.log(logs, step=global_step)

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
    # if accelerator.is_main_process:
    #     pipeline = StableDiffusionPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         unet=accelerator.unwrap_model(unet),
    #         text_encoder=accelerator.unwrap_model(text_encoder),
    #         revision=args.revision,
    #     )

    #     print("\n\nLora TRAINING DONE!\n\n")

    #     if args.output_format == "pt" or args.output_format == "both":
    #         save_lora_weight(pipeline.unet, args.output_dir + "/lora_weight.pt")
    #         if args.train_text_encoder:
    #             save_lora_weight(
    #                 pipeline.text_encoder,
    #                 args.output_dir + "/lora_weight.text_encoder.pt",
    #                 target_replace_module=["CLIPAttention"],
    #             )

        # if args.output_format == "safe" or args.output_format == "both":
        #     loras = {}
        #     if args.train_unet: 
        #         loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})
            
        #     print("Cross Attention is also updated!")
            
        #     # """ If updating only cross attention """
        #     # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})
            
        #     if args.train_text_encoder:
        #         loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

        #     # if loras != {}: 
        #     #     save_safeloras(loras, args.output_dir + "/lora_weight.safetensors")
        #     """
        #     ADOBE CONFIDENTIAL
        #     Copyright 2024 Adobe
        #     All Rights Reserved.
        #     NOTICE: All information contained herein is, and remains
        #     the property of Adobe and its suppliers, if any. The intellectual
        #     and technical concepts contained herein are proprietary to Adobe 
        #     and its suppliers and are protected by all applicable intellectual 
        #     property laws, including trade secret and copyright laws. 
        #     Dissemination of this information or reproduction of this material is 
        #     strictly forbidden unless prior written permission is obtained from Adobe.
        #     """
        #     torch.save(continuous_word_model.state_dict(), args.output_dir + "/continuous_word_mlp.pt")
        #     """end Adobe CONFIDENTIAL"""

        # if args.push_to_hub:
        #     repo.push_to_hub(
        #         commit_message="End of training",
        #         blocking=False,
        #         auto_lfs_prune=True,
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    controlnet_prompts = []
    prompts_file = open(args.controlnet_prompts_file)
    for line in prompts_file.readlines():
        prompt = str(line).strip() 
        controlnet_prompts.append(prompt)
    args.controlnet_prompts = controlnet_prompts 
    main(args)