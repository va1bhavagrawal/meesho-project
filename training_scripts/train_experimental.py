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
import os.path as osp 
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

TOKEN2ID = {
    "sks": 48136,
    "bnha": 49336, 
}


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

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import random
import re

from continuous_word_mlp import continuous_word_mlp
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb 

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

class ContinuousWordDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
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
        return self._length

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

            print(f"choosing from standard viewpoint only!, path is: {instance_img_path}")
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

            print(f"not using standard viewpoint, using controlnet augmentation instead!, path is: {controlnet_img_path}")
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
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


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
        "--seed", type=int, default=None, help="A seed for reproducible training."
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
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=5000,
        help="Total number of stage 1 training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=25000,
        help="Total number of stage 2 training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
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
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
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

    args.max_train_steps = args.stage1_steps + args.stage2_steps 
    args.output_dir = osp.join(args.output_dir, f"__{args.run_name}") 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.wandb and accelerator.is_main_process:
        wandb.login(key="6ab81b60046f7d7f6a7dca014a2fcaf4538ff14a") 
        if args.run_name is None: 
            wandb.init(project=args.project)
        else:
            wandb.init(project=args.project, name=args.run_name)
    
    if args.wandb:
        wandb_log_data = {}
        

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

    if args.seed is not None:
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

    args.learning_rate = (
        args.learning_rate
        * args.gradient_accumulation_steps
        * args.train_batch_size
        * accelerator.num_processes
    )

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

    args.learning_rate_text = (
        args.learning_rate_text  
        * args.gradient_accumulation_steps
        * args.train_batch_size
        * accelerator.num_processes
    )

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

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = ContinuousWordDataset(
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
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_schedulers = [get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    ) for optimizer in optimizers] 

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
    continuous_word_optimizer = torch.optim.Adam(continuous_word_model.parameters(), lr=1e-3)
    print("The current continuous MLP: {}".format(continuous_word_model))

    
    (
        unet,
        text_encoder,
        *optimizers,
        train_dataloader,
        continuous_word_model,
        continuous_word_optimizer,
    ) = accelerator.prepare(
        unet, text_encoder, *optimizers, train_dataloader, continuous_word_model, continuous_word_optimizer
    )

    lr_schedulers_ = [] 
    for idx in range(len(lr_schedulers)): 
        lr_schedulers_.append(accelerator.prepare(lr_schedulers[idx])) 

    lr_schedulers = lr_schedulers_ 
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
    
    vae.to(accelerator.device, dtype=weight_dtype)
    # FROM THE ORIGINAL CODE, SEEMS WRONG...
    # if not args.train_text_encoder:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)

    text_encoder.to(accelerator.device, dtype=weight_dtype) 
    unet.to(accelerator.device, dtype=weight_dtype) 


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    ddp_step = 0
    last_save = 0

    
    for epoch in range(args.num_train_epochs):
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
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

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
                input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight)  
                print("Stage 1 training: Disentangling object identity first")
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=False)
                encoder_hidden_states = text_encoder(batch["obj_ids"])[0]
            else:
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
                    text_encoder.module.get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=False)  

                    # performing the replacement on cold embeddings by a hot embedding -- allowed 
                    text_encoder.module.get_input_embeddings().weight[TOKEN2ID["sks"]] = mlp_emb[batch_idx] 

                    # appending to the encoder states 
                    encoder_hidden_states.append(text_encoder(batch_item.unsqueeze(0))[0].squeeze()) 


                encoder_hidden_states = torch.stack(encoder_hidden_states)  

                # replacing the text encoder input embeddings by the original ones, this time setting them to be HOT, this will be useful in case we choose to do textual inversion 
                text_encoder.module.get_input_embeddings().weight = torch.nn.Parameter(input_embeddings, requires_grad=True)   
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
            # TODO: see the error in the grad norm computation, gathering and logging 
            # if args.wandb:
            #     unet_grad_norm = [param.grad.norm() for param in unet.parameters() if param.grad is not None]
            #     if len(unet_grad_norm) == 0:
            #         unet_grad_norm = torch.tensor(0.0).to(accelerator.device) 
            #     else:
            #         unet_grad_norm = torch.norm(torch.stack(unet_grad_norm)) 
            #     mlp_grad_norm = [param.grad.norm() for param in continuous_word_model.parameters() if param.grad is not None]
            #     if len(mlp_grad_norm) == 0:
            #         mlp_grad_norm = torch.tensor(0.0).to(accelerator.device) 
            #     else:
            #         mlp_grad_norm = torch.norm(torch.stack(mlp_grad_norm)) 
            #     if args.train_text_encoder: 
            #         text_encoder_grad_norm = [param.grad.norm() for param in text_encoder.parameters() if param.grad is not None]
            #         if len(text_encoder_grad_norm) == 0:
            #             text_encoder_grad_norm = torch.tensor(0.0).to(accelerator.device) 
            #         else:
            #             text_encoder_grad_norm = torch.norm(torch.stack(text_encoder_grad_norm)) 
            #         all_grad_norms = torch.stack([unet_grad_norm, mlp_grad_norm, text_encoder_grad_norm]) 
            #     else:
            #         all_grad_norms = torch.stack([unet_grad_norm, mlp_grad_norm]) 

            #     # gathering all the norms at once to prevent excessive multi gpu communication 
            #     gathered_grad_norms = torch.mean(accelerator.gather(all_grad_norms), 0) 
            #     wandb_log_data["unet_grad_norm"] = gathered_grad_norms[0] 
            #     wandb_log_data["mlp_grad_norm"] =  gathered_grad_norms[1] 
            #     if args.train_text_encoder: 
            #         wandb_log_data["text_encoder_grad_norm"] = gathered_grad_norms[2]   


            # gradient clipping 
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters(), continuous_word_model.parameters())
                    if args.train_text_encoder
                    else itertools.chain(unet.parameters(), continuous_word_model.parameters())
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            continuous_word_optimizer.step()
            
            for optimizer in optimizers: 
                optimizer.step() 
            for lr_scheduler in lr_schedulers: 
                lr_scheduler.step()
            progress_bar.update(accelerator.num_processes * args.train_batch_size) 
            for optimizer in optimizers: 
                optimizer.zero_grad()
            continuous_word_optimizer.zero_grad()
            """end Adobe CONFIDENTIAL"""

            # since we have stepped, time to log weight norms!

            # calculate the weight norm for each of the trainable parameters 
            # if args.wandb:
            #     unet_norm = torch.norm(torch.stack([param for param in unet.parameters()])) 
            #     mlp_norm = torch.norm(torch.stack([param for param in continuous_word_model.parameters()])) 
            #     if args.train_text_encoder: 
            #         text_encoder_norm = torch.norm(torch.stack([param for param in text_encoder.parameters()])) 
            #         all_grad_norms = torch.stack([unet_norm, mlp_norm, text_encoder_norm]) 
            #     else:
            #         all_grad_norms = torch.stack([unet_norm, mlp_norm]) 
                
            #     gathered_norms = torch.mean(accelerator.gather(all_grad_norms), 0)
            #     wandb_log_data["unet_weight_norm"] = gathered_norms[0]
            #     wandb_log_data["mlp_weight_norm"] = gathered_norms[1]
            #     if args.train_text_encoder: 
            #         wandb_log_data["text_encoder_weight_norm"] = gathered_norms[2] 

            
            global_step += accelerator.num_processes * args.train_batch_size  
            ddp_step += 1
            if args.wandb:
                wandb_log_data["global_step"] = global_step 

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.save_steps and global_step - last_save >= args.save_steps:
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

                        filename_unet = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                        )
                        filename_text_encoder = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                        print(f"save weights {filename_unet}, {filename_text_encoder}")
                        save_lora_weight(pipeline.unet, filename_unet)
                        
                        if args.output_format == "safe" or args.output_format == "both":
                            loras = {}
                            loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})

                            print("Cross Attention is also updated!")

                            # """ If updating only cross attention """
                            # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})

                            if args.train_text_encoder:
                                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

                            save_safeloras(loras, f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.safetensors")
                            
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
                            torch.save(continuous_word_model.state_dict(), f"{args.output_dir}/mlp{epoch}_s{global_step}.pt")
                            """End Adobe CONFIDENTIAL"""
                        
                        
                        if args.train_text_encoder:
                            save_lora_weight(
                                pipeline.text_encoder,
                                filename_text_encoder,
                                target_replace_module=["CLIPAttention"],
                            )


                        last_save = global_step

            loss = loss.detach()
            gathered_loss = torch.mean(accelerator.gather(loss), 0)
            if args.wandb:
                wandb_log_data["loss"] = gathered_loss
                # wandb_log_data["lr"] =  lr_scheduler.get_last_lr()[0]

            if args.wandb and accelerator.is_main_process and ddp_step % 5 == 0: 
                # finally logging!
                wandb.log(wandb_log_data) 

            logs = {"loss": gathered_loss, "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    wandb.finish() 

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\TRAINING DONE!\n\n")

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
            loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn", "Attention", "GEGLU"})
            
            print("Cross Attention is also updated!")
            
            # """ If updating only cross attention """
            # loras["unet"] = (pipeline.unet, {"CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"})
            
            if args.train_text_encoder:
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

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