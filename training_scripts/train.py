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

# from metrics import MetricEvaluator 


TOKEN2ID = {
    "sks": 48136,
    "bnha": 49336, 
}
DEBUG = False 
BS = 4 
SAVE_STEPS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000] 
VLOG_STEPS = copy.deepcopy(SAVE_STEPS)  
# VLOG_STEPS = [32, 64] 


from accelerate import Accelerator
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

from continuous_word_mlp import continuous_word_mlp, HotEmbedding 
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


def infer(args, step, accelerator, unet, scheduler, vae, text_encoder, mlp, use_sks, bnha_embed=None):  
    root_save_path = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}") 
    with torch.no_grad(): 
        vae.to(accelerator.device) 
        # the list of videos 
        # each item in the list is the video of a prompt at different viewpoints, or just random generations if use_sks=False  
        accelerator.print(f"performing inference...") 
        videos = {}  
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer", 
        ) 

        if not use_sks: 
            prompts_dataset = PromptDataset(use_sks=use_sks, num_samples=6)  
        else: 
            prompts_dataset = PromptDataset(use_sks=use_sks, num_samples=24)  

        if args.textual_inv: 
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = bnha_embed(0)  

        n_prompts_per_azimuth = len(prompts_dataset.subjects) * len(prompts_dataset.template_prompts) 
        encoder_hidden_states = torch.zeros((prompts_dataset.num_samples * n_prompts_per_azimuth, 77, 1024)).to(accelerator.device).contiguous()  

        accelerator.print(f"collecting the encoder hidden states...") 
        for azimuth in range(prompts_dataset.num_samples): 
            if azimuth % accelerator.num_processes == accelerator.process_index: 
                normalized_azimuth = azimuth / prompts_dataset.num_samples 
                sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(normalized_azimuth)), torch.cos(2 * torch.pi * torch.tensor(normalized_azimuth))]).to(accelerator.device) 
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp(sincos.unsqueeze(0)) 
                tokens = tokenizer(
                    prompts_dataset.prompts, 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length,
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids 
                text_encoder_outputs = text_encoder(tokens.to(accelerator.device))[0]   
                encoder_hidden_states[azimuth * n_prompts_per_azimuth : (azimuth + 1) * n_prompts_per_azimuth] = text_encoder_outputs  
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
        accelerator.print(f"starting generation...")  
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
            os.makedirs(f"../gpu_imgs/{accelerator.process_index}", exist_ok=True) 
            for idx, image in zip(ids, images):  
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image * 255).to(torch.uint8) 
                image = image.cpu().numpy() 
                image = Image.fromarray(image) 
                azimuth = idx // n_prompts_per_azimuth 
                prompt_idx = idx % n_prompts_per_azimuth 
                prompt = prompts_dataset.prompts[prompt_idx] 
                prompt_filename = "_".join(prompt.split()) 
                save_path = osp.join(root_save_path, prompt_filename) 
                os.makedirs(save_path, exist_ok=True) 
                save_path = osp.join(root_save_path, prompt_filename, f"{str(int(azimuth.item())).zfill(3)}.jpg")   
                image.save(save_path) 
            
        accelerator.wait_for_everyone() 
        vae = vae.to(torch.device(f"cpu")) 

        videos = {} 
        for prompt_filename in os.listdir(root_save_path): 
            for img_name in os.listdir(osp.join(root_save_path, prompt_filename)): 
                img_path = osp.join(root_save_path, prompt_filename, img_name) 
                if prompt_filename not in videos.keys(): 
                    videos[prompt_filename] = [] 
                videos[prompt_filename].append(Image.open(img_path))   

        return videos
                # image = image.cpu().numpy()  
                # image = np.transpose(image, (1, 2, 0)) 
                # image = Image.fromarray(image) 
                # image.save(osp.join(f"../gpu_imgs/{accelerator.process_index}", f"{str(int(idx.item())).zfill(3)}.jpg")) 

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


class ContinuousWordDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        args, 
        controlnet_prompts,
        instance_data_root,
        controlnet_data_dir,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
    ):
        self.args = args 
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize
        self.controlnet_prompts = controlnet_prompts

        self.instance_images_path = []
        for cur_root in glob.glob(instance_data_root):
            self.instance_images_path += [cur_dir for cur_dir in Path(cur_root).iterdir() if '.jpg' in str(cur_dir)]
        
        self.controlnet_images_path = []
        for cur_root in glob.glob(controlnet_data_dir):
            self.controlnet_images_path += [cur_dir for cur_dir in Path(cur_root).iterdir() if '.jpg' in str(cur_dir)]
            
        print("Length of images used for training {}".format(len(self.instance_images_path)))
        
        self.num_instance_images = len(self.instance_images_path)
        self.num_controlnet_images = len(self.controlnet_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        img_transforms = []

        if resize:
            img_transforms.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        
        # rare_token_obj is for the object identity, rare token is continuous mlp placeholder
        self.rare_token_obj = 'bnha'
        self.rare_token = 'sks'

    def __len__(self):
        return args.max_train_steps  

    def __getitem__(self, index):
        example = {}

        instance_img_path = self.instance_images_path[index % self.num_instance_images]
        angle = float(str(instance_img_path).split("/")[-1].split("_.jpg")[0]) 
        example["scaler"] = angle 
        """Maintain the same sentence for object tokens"""
        if index % 5 != 0:  
            obj_caption = f'a bnha {args.subject}'
            """IMPORTANT: Remove in a white background if it makes the results worse"""
            caption = f'a sks photo of a bnha {args.subject} in front of a dark background'

            example["obj_prompt_ids"] = self.tokenizer(
                obj_caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

            example["instance_prompt_ids"] = self.tokenizer(
                caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

            # print(f"choosing from standard viewpoint only!, path is: {instance_img_path}")
            instance_img = Image.open(
                instance_img_path 
            )

        else:
            controlnet_img_paths = list(self.controlnet_images_path)
            controlnet_img_paths = [str(img_path) for img_path in controlnet_img_paths if str(img_path).find(str(angle)) != -1]
            controlnet_img_path = random.choice(controlnet_img_paths)
            assert controlnet_img_path.find(str(example["scaler"])) != -1 
            prompt_idx = int(controlnet_img_path.split("___prompt")[1].split(".jpg")[0])
            assert prompt_idx < len(self.controlnet_prompts) 
            img_desc = self.controlnet_prompts[prompt_idx]
            obj_caption = img_desc
            caption = 'a sks photo of ' + img_desc

            example["obj_prompt_ids"] = self.tokenizer(
                obj_caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

            example["instance_prompt_ids"] = self.tokenizer(
                caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

            # print(f"not using standard viewpoint, using controlnet augmentation instead!, path is: {controlnet_img_path}")
            instance_img = Image.open(controlnet_img_path) 

        if not instance_img.mode == "RGB": 
            instance_img = instance_img.convert("RGB") 
        example["instance_images"] = self.image_transforms(instance_img)  
            

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

"""end Adobe CONFIDENTIAL"""

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
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
        default=5,
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
        "--stage1_steps",
        type=int,
        default=5000,
        help="Number of steps for stage 1 training", 
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=25000,
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
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
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


def main(args, controlnet_prompts):

    args.output_dir = osp.join(args.output_dir, f"__{args.run_name}") 
    args.max_train_steps = args.stage1_steps + args.stage2_steps 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    assert accelerator.num_processes * args.train_batch_size == BS, f"{accelerator.num_processes = }, {args.train_batch_size = }" 

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

    set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        assert cur_class_images == args.num_class_images 

    # Handle the repository creation
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
    optimizers = [] 
    if args.train_unet: 
        optimizer_unet = optimizer_class(
            itertools.chain(*unet_lora_params), 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizers.append(optimizer_unet) 

    if args.train_text_encoder: 
        optimizer_text_encoder = optimizer_class(
            itertools.chain(*text_encoder_lora_params),  
            lr=args.learning_rate_text,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizers.append(optimizer_text_encoder) 

    if args.textual_inv: 
        bnha_embed = torch.clone(text_encoder.get_input_embeddings().weight[TOKEN2ID["bnha"]])  
        bnha_embed = HotEmbedding(bnha_embed).to(accelerator.device) 
        optimizer_bnha = optimizer_class(
            bnha_embed.parameters(),  
            lr=args.learning_rate_emb,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizers.append(optimizer_bnha) 


    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = ContinuousWordDataset(
        args=args,
        controlnet_prompts=controlnet_prompts,
        instance_data_root=args.instance_data_dir,
        controlnet_data_dir = args.controlnet_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
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
        input_ids = [example["instance_prompt_ids"] for example in examples]
        obj_ids = [example["obj_prompt_ids"] for example in examples]
        pixel_values = []
        for example in examples:
            pixel_values.append(example["instance_images"])

        
        """Adding the scaler of the embedding into the batch"""
        scalers = torch.Tensor([example["scaler"] for example in examples])

        
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            obj_ids += [example["class_prompt_ids"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        obj_ids = tokenizer.pad(
            {"input_ids": obj_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "obj_ids": obj_ids,
            "pixel_values": pixel_values,
            "scalers": scalers,
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
    pos_size = 2
    continuous_word_model = continuous_word_mlp(input_size=pos_size, output_size=1024)
    continuous_word_optimizer = torch.optim.Adam(continuous_word_model.parameters(), lr=args.learning_rate_mlp) 
    optimizers.append(continuous_word_optimizer) 
    print("The current continuous MLP: {}".format(continuous_word_model))
    
    
    # if args.train_text_encoder:
    #     (
    #         unet,
    #         text_encoder,
    #         optimizer_unet,
    #         optimizer_text_encoder,
    #         train_dataloader,
    #         lr_scheduler,
    #         continuous_word_model,
    #         continuous_word_optimizer
    #     ) = accelerator.prepare(
    #         unet, text_encoder, optimizer_unet, optimizer_text_encoder, train_dataloader, lr_scheduler, continuous_word_model, continuous_word_optimizer
    #     )
    # else:
    #     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, optimizer, train_dataloader, lr_scheduler
    #     )
    unet, text_encoder, continuous_word_model, train_dataloader = accelerator.prepare(unet, text_encoder, continuous_word_model, train_dataloader)  
    optimizers_ = [] 
    for optimizer in optimizers: 
        optimizer = accelerator.prepare(optimizer) 
        optimizers_.append(optimizer) 
    if args.textual_inv: 
        bnha_embed = accelerator.prepare(bnha_embed) 
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

    for step, batch in enumerate(train_dataloader):
        if args.wandb:
            wandb_log_data = {}
        force_wandb_log = False 
        # Convert images to latent space
        vae.to(accelerator.device, dtype=weight_dtype)
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * 0.18215
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

        # Get the text embedding for conditioning
        if global_step <= args.stage1_steps: 
            progress_bar.set_description(f"stage 1: ")
            input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight)  
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=False)
            if args.textual_inv: 
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = bnha_embed(0)  
            encoder_hidden_states = text_encoder(batch["obj_ids"])[0]
        else:
            progress_bar.set_description(f"stage 2: ")
            input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight)  
            # print("Stage 2 training: Learning Continuous Word MLP")
            # # normalization of the scalers
            # p = torch.Tensor((batch["scalers"])/(2 * math.pi))
            
            # # Positional Encoding
            # x = torch.Tensor(
            #     [torch.sin(2 * torch.pi * p), torch.cos(2 * torch.pi * p)]).cuda()

            # mlp_emb = continuous_word_model(torch.unsqueeze(x, dim=0)).squeeze(0)
            # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=False)
            
            # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[batch["input_ids"][0][2]] = mlp_emb
            # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            p = torch.Tensor(batch["scalers"] / (2 * math.pi)) 
            p = p.unsqueeze(-1)
            p = p.repeat(1, 2)
            p[:, 0] = torch.sin(2 * torch.pi * p[:, 0]) 
            p[:, 1] = torch.cos(2 * torch.pi * p[:, 1])

            # getting the embeddings from the mlp
            mlp_emb = continuous_word_model(p) 

            # checking what token is actually used for pose conditioning 
            assert batch["input_ids"][0][2] == TOKEN2ID["sks"]  

            # replacing the input embedding for sks by the mlp for each batch item, and then getting the output embeddings of the text encoder 
            # must run a for loop here, first changing the input embeddings of the text encoder for each 
            encoder_hidden_states = []
            input_ids, input_ids_prior = torch.chunk(batch["input_ids"], 2, dim=0) 


            for batch_idx, batch_item in enumerate(input_ids): 
                # replacing the text encoder input embeddings by the original ones and setting them to be COLD -- to enable replacement by a hot embedding  
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=False)  

                # performing the replacement on cold embeddings by a hot embedding -- allowed 
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp_emb[batch_idx] 
                if args.textual_inv: 
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID["bnha"]] = bnha_embed(0)  


                # appending to the encoder states 
                encoder_hidden_states.append(text_encoder(batch_item.unsqueeze(0))[0].squeeze()) 


            encoder_hidden_states = torch.stack(encoder_hidden_states)  

            # replacing the text encoder input embeddings by the original ones, this time setting them to be HOT, this will be useful in case we choose to do textual inversion 
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=True)   
            encoder_hidden_states_prior = text_encoder(input_ids_prior)[0] 
            assert encoder_hidden_states_prior.shape == encoder_hidden_states.shape 
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_prior], dim=0)

        """End Adobe CONFIDENTIAL"""


        # Predict the noise residual
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

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        # everytime the continuous word mlp must receive gradients 
        if DEBUG: 
            with torch.no_grad(): 
                if global_step > args.stage1_steps:  
                    check_mlp_params = [p for p in continuous_word_model.parameters() if p.grad is None] 
                    assert len(check_mlp_params) == 0 
                    del check_mlp_params 
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
                    bnha_grad_norm = [torch.linalg.norm(param.grad) for param in bnha_embed.parameters() if param.grad is not None] 
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
                curr = 1 
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
            parmas_to_clip = params_to_clip + list(itertools.chain(continuous_word_model.parameters()))  
            if args.train_unet: 
                params_to_clip = parmas_to_clip + list(itertools.chain(unet.parameters()))  
            if args.train_text_encoder: 
                params_to_clip = parmas_to_clip + list(itertools.chain(text_encoder.parameters()))  
            if args.textual_inv: 
                params_to_clip = params_to_clip + list(itertools.chain(bnha_embed.parameters())) 
            # params_to_clip = (
            #     itertools.chain(unet.parameters(), text_encoder.parameters(), continuous_word_model.parameters())
            #     if args.train_text_encoder
            #     else itertools.chain(unet.parameters(), continuous_word_model.parameters())
            # )
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        if DEBUG: 
            with torch.no_grad(): 
                if global_step > args.stage1_steps: 
                    mlp_before = copy.deepcopy([p for p in continuous_word_model.parameters()])  
                if args.train_unet: 
                    unet_before = copy.deepcopy([p for p in unet.parameters()]) 
                if args.train_text_encoder: 
                    text_encoder_before = copy.deepcopy([p for p in text_encoder.parameters()]) 
                if args.textual_inv: 
                    bnha_before = copy.deepcopy([p for p in bnha_embed.parameters()]) 

        for optimizer in optimizers: 
            optimizer.step() 

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
                    bnha_norm = [torch.linalg.norm(param) for param in bnha_embed.parameters() if param.grad is not None] 
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
                curr = 1 
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
            with torch.no_grad(): 
                if global_step > args.stage1_steps: 
                    mlp_after = copy.deepcopy([p for p in continuous_word_model.parameters()])  
                if args.train_unet: 
                    unet_after = copy.deepcopy([p for p in unet.parameters()]) 
                if args.train_text_encoder: 
                    text_encoder_after = copy.deepcopy([p for p in text_encoder.parameters()]) 
                if args.textual_inv: 
                    bnha_after = copy.deepcopy([p for p in bnha_embed.parameters()]) 

                if global_step > args.stage1_steps: 
                    mlp_after = [p1 - p2 for p1, p2 in zip(mlp_before, mlp_after)] 
                    del mlp_before 
                if args.train_unet: 
                    unet_after = [p1 - p2 for p1, p2 in zip(unet_before, unet_after)]  
                    del unet_before 
                if args.train_text_encoder: 
                    text_encoder_after = [p1 - p2 for p1, p2 in zip(text_encoder_before, text_encoder_after)]  
                    del text_encoder_before
                if args.textual_inv: 
                    bnha_after = [p1 - p2 for p1, p2 in zip(bnha_before, bnha_after)]   
                    del bnha_before 

                if global_step > args.stage1_steps: 
                    for p_diff in mlp_after:  
                        if torch.sum(p_diff): 
                            change = True 
                            break 
                    assert change 

                change = False 
                if args.train_unet: 
                    for p_diff in unet_after:  
                        if torch.sum(p_diff): 
                            change = True 
                            break 
                    assert change 

                change = False 
                if args.train_text_encoder: 
                    for p_diff in text_encoder_after:  
                        if torch.sum(p_diff): 
                            change = True 
                            break 
                    assert change 
            
                change = False 
                if args.textual_inv:  
                    for p_diff in bnha_after:  
                        if torch.sum(p_diff): 
                            change = True 
                            break 
                    assert change 


        progress_bar.update(accelerator.num_processes * args.train_batch_size) 

        # optimizer_unet.zero_grad()
        # optimizer_text_encoder.zero_grad()
        # continuous_word_optimizer.zero_grad()
        for optimizer in optimizers: 
            optimizer.zero_grad() 

        """end Adobe CONFIDENTIAL"""

        # since we have stepped, time to log weight norms!

        global_step += accelerator.num_processes * args.train_batch_size  
        ddp_step += 1

        if args.online_inference and len(VLOG_STEPS) > 0 and global_step >= VLOG_STEPS[0]:  
            step = VLOG_STEPS[0] 
            VLOG_STEPS.pop(0) 
            if global_step <= args.stage1_steps:  
                use_sks = False 
            else:
                use_sks = True 
            if args.textual_inv: 
                videos = infer(args, step, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, use_sks, bnha_embed) 
            else: 
                videos = infer(args, step, accelerator, unet, noise_scheduler, vae, text_encoder, continuous_word_model, use_sks) 

            if accelerator.is_main_process: 
                for key, value in videos.items():  
                    save_path = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step}", key) 
                    os.makedirs(save_path, exist_ok=True)  
                    save_path = osp.join(save_path, key + ".gif") 
                    create_gif(value, save_path, duration=1) 
                    if args.wandb: 
                        prompt = " ".join(key.split("_"))
                        wandb_log_data[prompt] = wandb.Video(save_path)    

                    force_wandb_log = True 
                
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

                # metrics computation 
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
                        torch.save(continuous_word_model.state_dict(), f"{args.output_dir}/mlp_{global_step}.pt")
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
        if args.wandb and ddp_step % args.log_every == 0:
            wandb_log_data["loss"] = gathered_loss

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
        prompt = "a" + prompt[1:]
        controlnet_prompts.append(prompt)
    main(args, controlnet_prompts)