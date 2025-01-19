# from pipeline_stable_diffusion_3_online_vis_attn import * 
# from pipeline_sd3_general import * 
from accelerate.utils import set_seed 
from utils import * 


import argparse
import copy
import itertools
import logging
import math
import os
import os.path as osp 
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import pickle 
import matplotlib.pyplot as plt 

import numpy as np
import torch
import torch.nn as nn 
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import (
	check_min_version,
	convert_unet_state_dict_to_peft,
	is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from custom_attention_processor2 import patch_custom_attention 


#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import itertools
import json
import logging
import math
import os
import os.path as osp 
import copy 
import matplotlib.pyplot as plt 
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from utils import * 

# from datasets import EveryPoseEveryThingDataset 
from continuous_word_mlp import GoodPoseEmbedding 

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers

from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	DPMSolverMultistepScheduler,
	EDMEulerScheduler,
	EulerDiscreteScheduler,
	# StableDiffusionXLPipeline,
	UNet2DConditionModel,
)
from pipeline_stable_diffusion_xl import SDXLWithCALL 
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
	check_min_version,
	convert_all_state_dict_to_peft,
	convert_state_dict_to_diffusers,
	convert_state_dict_to_kohya,
	convert_unet_state_dict_to_peft,
	is_peft_version,
	is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

##################  
VISUALIZE_ATTN = False 
TIMESTEPS_TO_VIS_ATTN = range(0, 28, 3)   
VIS_ATTN_LAYERS = range(24) 
ZERO_T5 = True 
ZERO_CLIP1 = True 
ZERO_CLIP2 = True 
ZERO_POOLED = True 
ZERO_TEXT_TILL_TIME = 5 
NUM_SAMPLES = 8   


# ROOT_CKPTS_DIR = "/ssd_scratch/vaibhav/ckpts"
ROOT_CKPTS_DIR = "../ckpts/"
WHICH_RUN = "sdxl1024_1e-4_1e-3_CALL_stage2_from60000_cn7"  
WHICH_STEP = "100000"   
WHAT = ""
MAX_BATCH_SIZE = 100 
MAX_SUBJECTS = 100 
INFER_BATCH_SIZE = 4        
FIXED_SEED = False 
OUTPUT_DIR = "output" 
##################  


def online_inference(pipeline, tmp_dir, accelerator, conditioning_kwargs={}):  
	if conditioning_kwargs != {}: 
		# special_encoder_part1 = conditioning_kwargs["special_encoder_part1"] 
		# special_encoder_part2_one = conditioning_kwargs["special_encoder_part2_one"] 
		# special_encoder_part2_two = conditioning_kwargs["special_encoder_part2_two"] 
		# special_encoder_part2_three = conditioning_kwargs["special_encoder_part2_three"] 
		special_encoder = conditioning_kwargs["special_encoder"] 
		special_encoder_two = conditioning_kwargs["special_encoder_two"] 
		num_samples = conditioning_kwargs["num_samples"] 
		special_tokens_ints_one = conditioning_kwargs["special_tokens_ints_one"] 
		special_tokens_ints_two = conditioning_kwargs["special_tokens_ints_two"] 

		# special_encoder_part1.eval() 
		# special_encoder_part2_one.eval() 
		# special_encoder_part2_two.eval() 
		# special_encoder_part2_three.eval() 
	else: 
		num_samples = 1 
	if accelerator.is_main_process: 
		if osp.exists(tmp_dir): 
			shutil.rmtree(tmp_dir) 

	scenes_data = [
		[ # the last one in this list contains the prompt and other meta details 
			{
				"name": "jeep", 
				"theta": np.linspace(0, 2 * np.pi, num_samples + 1)[:-1],   
				"bbox": [0.00, 0.50, 0.50, 1.00], 
				"x": -5.0,
				"y": +0.00, 
			}, 
			{
				"prompt": "a photo of PLACEHOLDER in a rocky terrain"  
			} 
		], 
		[ # the last one in this list contains the prompt and other meta details 
			{
				"name": "bicycle", 
				"theta": np.linspace(0, 2 * np.pi, num_samples + 1)[:-1],   
				"bbox": [0.25, 0.25, 0.75, 0.75], 
				"x": -5.0,
				"y": +0.00, 
			}, 
			{
				"prompt": "a photo of PLACEHOLDER in a backyard"  
			} 
		], 
		[ # the last one in this list contains the prompt and other meta details 
			{
				"name": "sedan", 
				"theta": np.linspace(0, 2 * np.pi, num_samples + 1)[:-1],   
				"bbox": [0.25, 0.25, 0.75, 0.75], 
				"x": -5.0,
				"y": +0.00, 
			}, 
			{
				"prompt": "a photo of PLACEHOLDER in a city street"  
			} 
		], 
		[ # the last one in this list contains the prompt and other meta details 
			{
				"name": "suv", 
				"theta": np.linspace(0, 2 * np.pi, num_samples + 1)[:-1],   
				"bbox": [0.50, 0.50, 1.00, 1.00], 
				"x": -5.0,
				"y": +0.00, 
			}, 
			{
				"prompt": "a photo of PLACEHOLDER on a highway"  
			} 
		], 
		[ # the last one in this list contains the prompt and other meta details 
			{
				"name": "ship", 
				"theta": np.linspace(0, 2 * np.pi, num_samples + 1)[:-1],   
				"bbox": [0.25, 0.50, 0.75, 1.00], 
				"x": -5.0,
				"y": +0.00, 
			}, 
			{
				"prompt": "a photo of PLACEHOLDER in a calm sea at sunset"  
			} 
		], 
	] 

	text_encoder_one = pipeline.text_encoder 
	text_encoder_two = pipeline.text_encoder_2 

	latents_store = torch.load("latents.pt") 
	for scene_idx, scene in enumerate(scenes_data): 
		# latents = random.choice(latents_store) 
		random_idx = torch.randint(0, len(latents_store), (1,)).float().to(accelerator.device)  
		random_idx = torch.mean(accelerator.gather(random_idx)).int()  
		latents = latents_store[random_idx] 
		prompts = [] 
		subjects_data = scene[:-1] 
		metadata = scene[-1] 
		subjects_info = [[{
			"bbox": subject_data["bbox"], 
		} for subject_data in scene[:-1]]] * num_samples   
		for orientation_idx in range(num_samples): 
			placeholder_text = "" 
			for subject_idx, subject_data in enumerate(subjects_data): 
				x = subject_data["x"] 
				y = subject_data["y"] 
				theta = subject_data["theta"][orientation_idx] 
				token_one = special_tokens_ints_one[orientation_idx * MAX_SUBJECTS + subject_idx] 
				token_two = special_tokens_ints_two[orientation_idx * MAX_SUBJECTS + subject_idx] 
				# token_two = special_tokens_ints_two[orientation_idx * MAX_SUBJECTS + subject_idx] 

				x_t = torch.tensor(x).unsqueeze(0).unsqueeze(-1).to(accelerator.device).float()  
				y_t = torch.tensor(y).unsqueeze(0).unsqueeze(-1).to(accelerator.device).float()  
				theta_t = torch.tensor(theta).unsqueeze(0).unsqueeze(-1).to(accelerator.device).float()   

				# intermediate_embedding = special_encoder_part1(x_t, y_t, theta_t)  
				# special_embedding_one = special_encoder_part2_one(intermediate_embedding) 
				# special_embedding_two = special_encoder_part2_two(intermediate_embedding) 
				# special_embedding_three = special_encoder_part2_three(intermediate_embedding) 
				special_embedding = special_encoder(theta_t / (2 * torch.pi)) 
				special_embedding_two = special_encoder_two(theta_t / (2 * torch.pi)) 

				# text_encoder_one.get_input_embeddings().weight[token_one] = special_embedding_one  
				# text_encoder_two.get_input_embeddings().weight[token_two] = special_embedding_two  
				# text_encoder_three.get_input_embeddings().weight[token_three] = special_embedding_three 
				text_encoder_one.get_input_embeddings().weight[token_one] = special_embedding 
				text_encoder_two.get_input_embeddings().weight[token_two] = special_embedding_two 

				prompt = metadata["prompt"] 
				# replace PLACEHOLDER with the subject names  
				subject_name = subject_data["name"] 
				if subject_idx == 0: 
					placeholder_text = placeholder_text + f"<special_token_{orientation_idx}_{subject_idx}> {subject_name}"  
				else: 
					placeholder_text = placeholder_text + f" and <special_token_{orientation_idx}_{subject_idx}> {subject_name}"  

			prompt = prompt.replace("PLACEHOLDER", placeholder_text) 
			prompts.append(prompt) 
		print(f"{prompts = }")
		placeholder_text_wo_special_tokens = "" 
		for subject_idx, subject_data in enumerate(subjects_data): 
			placeholder_text_wo_special_tokens = placeholder_text_wo_special_tokens + f"{subject_data['name']} and " 
		prompt_filename = sanitize_filename(scene[-1]["prompt"].replace("PLACEHOLDER", placeholder_text_wo_special_tokens))    

		prompt_ids = list(range(len(prompts)))  
		save_dir = osp.join(tmp_dir, prompt_filename) 
		if accelerator.is_main_process: 
			os.makedirs(save_dir, exist_ok=True) 
		with accelerator.split_between_processes(prompt_ids) as gpu_prompt_ids:  
			batch_size = 2     
			print(f"GPU {accelerator.process_index} is assigned {gpu_prompt_ids} / {prompt_ids}")  
			for start_idx in range(0, len(gpu_prompt_ids) - 1, batch_size):  
				end_idx = min(len(gpu_prompt_ids), start_idx + batch_size) 
				gpu_prompt_ids_batch = gpu_prompt_ids[start_idx : end_idx]  
				subjects_info_batch = [subjects_info[i] for i in gpu_prompt_ids_batch] 
				print(f"GPU {accelerator.process_index} is doing {gpu_prompt_ids_batch}")
				gpu_prompts_batch = [prompts[prompt_idx] for prompt_idx in gpu_prompt_ids_batch] 
				# images = pipeline(gpu_prompts_batch, num_inference_steps).images  
				tokens_batch = tokenize_prompt(pipeline.tokenizer, gpu_prompts_batch) 
				for batch_idx in range(len(gpu_prompt_ids_batch)): 
					for subject_idx in range(len(subjects_data)):  
						# TODO remove the hardcoded number and write the logic 
						subjects_info_batch[batch_idx][subject_idx]["special_token_idx"] = 4 + 2 * subject_idx   
						subjects_info_batch[batch_idx][subject_idx]["subject_token_idx"] = 4 + 2 * subject_idx + 1  
				images = pipeline(prompt=gpu_prompts_batch, latents=latents.unsqueeze(0).repeat(len(gpu_prompts_batch), 1, 1, 1), subjects_info=subjects_info_batch).images  
				for prompt_idx, image in zip(gpu_prompt_ids_batch, images): 
					save_dir_orientation = osp.join(save_dir, f"{str(prompt_idx).zfill(3)}") 
					os.makedirs(save_dir_orientation, exist_ok=True) 
					save_path = osp.join(save_dir_orientation, f"img.jpg") 
					image.save(save_path)  
				del images 
				torch.cuda.empty_cache() 
		accelerator.wait_for_everyone() 
# road_prompt_list = [
#     "A photo of PLACEHOLDER in front of the Taj",
#     "A photo of PLACEHOLDER on the streets of Venice, with the sun setting in the background",
#     "A photo of PLACEHOLDER in front of the leaning tower of Pisa in Italy",
#     "A photo of PLACEHOLDER in a modern city street surrounded by towering skyscrapers and neon lights",
#     "A photo of PLACEHOLDER in an ancient Greek temple ruin, with broken columns and weathered stone steps",
#     "A photo of PLACEHOLDER in a field of dandelions, with snowy mountain peaks in the distance",
#     "A photo of PLACEHOLDER in a rustic village with cobblestone streets and small houses",
#     "A photo of PLACEHOLDER on a winding country road with green fields, trees, and distant mountains under a sunny sky",
#     "A photo of PLACEHOLDER in front of a serene waterfall with trees scattered around the region, and stones scattered in the water",
#     "A photo of PLACEHOLDER on a sandy desert road with dunes and a vast, open sky above",
#     "A photo of PLACEHOLDER on a bridge overlooking a river with mountains in the background",
#     "A photo of PLACEHOLDER on a dirt path in a dense forest with sunbeams filtering through the trees",
#     "A photo of PLACEHOLDER on a coastal road with cliffs overlooking the ocean",
#     "A photo of PLACEHOLDER in front of a historical castle with high stone walls and flags flying in the breeze",
#     "A photo of PLACEHOLDER in front of an amusement park with bright lights and ferris wheels in the background"
# ]

# water_prompt_list = [
#     "A photo of PLACEHOLDER on still waters under a cloudy sky, mountains visible in the distant horizon",
#     "A photo of PLACEHOLDER floating on a misty lake, surrounded by calm waters and serene, foggy atmosphere",
#     "A photo of PLACEHOLDER in the vast sea, with a clear blue sky and a few fluffy clouds",
#     "A photo of PLACEHOLDER in the middle of a stormy ocean, with dark clouds and crashing waves",
#     "A photo of PLACEHOLDER in a calm lake with lily pads and reeds growing near the shoreline",
#     "A photo of PLACEHOLDER on a river running through a dense jungle with vibrant green foliage",
#     "A photo of PLACEHOLDER in a mountain lake surrounded by pine trees and snow-capped peaks",
#     "A photo of PLACEHOLDER floating in a lagoon with tropical fish and coral visible beneath the water",
#     "A photo of PLACEHOLDER on a frozen lake with a snowy landscape surrounding it",
#     "A photo of PLACEHOLDER on a serene river at dusk, with reflections of the sunset on the water",
#     "A photo of PLACEHOLDER in the middle of a vast marshland with tall grasses and migratory birds flying overhead",
#     "A photo of PLACEHOLDER near a small waterfall cascading into a clear pool in a rocky area",
#     "A photo of PLACEHOLDER on a bay with large rock formations jutting out of the water",
#     "A photo of PLACEHOLDER in a turquoise sea with gentle waves and distant islands on the horizon",
#     "A photo of PLACEHOLDER in a narrow canal in an old European city, with historic buildings lining the waterway"
# ]

# indoor_prompt_list = [
#     "A photo of PLACEHOLDER in a modern living room setting with painted walls and glass windows",
#     "A photo of PLACEHOLDER in a minimalist living room",
#     "A photo of PLACEHOLDER in a cozy library with shelves filled with books and warm lighting",
#     "A photo of PLACEHOLDER in a high-tech office with large windows and a city view",
#     "A photo of PLACEHOLDER in an art studio with canvas paintings and art supplies scattered around",
#     "A photo of PLACEHOLDER in a rustic kitchen with wooden cabinets and a stone countertop",
#     "A photo of PLACEHOLDER in a lavish living room with elegant decor and soft lighting",
#     "A photo of PLACEHOLDER in a large dining hall with chandeliers and long tables",
#     "A photo of PLACEHOLDER in a traditional Japanese tatami room with sliding paper doors",
#     "A photo of PLACEHOLDER in a well-equipped gym with weights and fitness machines",
#     "A photo of PLACEHOLDER in a music studio with soundproof walls and musical instruments",
#     "A photo of PLACEHOLDER in a sunlit greenhouse filled with tropical plants",
#     "A photo of PLACEHOLDER in a children's playroom with colorful toys and posters on the walls",
#     "A photo of PLACEHOLDER in an underground wine cellar with wooden barrels and dim lighting",
#     "A photo of PLACEHOLDER in a cozy reading nook with a soft armchair and a small lamp"
# ]
road_prompt_list = [
	"A photo of PLACEHOLDER in Venice", 
	"A photo of PLACEHOLDER in a forest",  
	"A photo of PLACEHOLDER on a highway", 
]

water_prompt_list = [
	"A photo of PLACEHOLDER in the ocean",  
	"A photo of PLACEHOLDER in a calm lake",  
	"A photo of PLACEHOLDER in a river at sunset",  
]

indoor_prompt_list = [
	"A photo of PLACEHOLDER in a modern living room",  
	"A photo of PLACEHOLDER in a sunlit greenhouse",  
	"A photo of PLACEHOLDER in a library",  
]

# Expanded subjects dictionary
subjects_and_prompts = {
	"sedan": road_prompt_list, 
	"boat": water_prompt_list, 
	"dolphin": water_prompt_list, 
	"horse": road_prompt_list, 
	"jeep": road_prompt_list, 
	"ship": water_prompt_list, 
	"sofa": indoor_prompt_list, 
	"suv": road_prompt_list, 
	"teddy": indoor_prompt_list, 
	"chair": indoor_prompt_list, 
	"tractor": road_prompt_list, 
	# New subjects
	"motorcycle": road_prompt_list,
	"submarine": water_prompt_list,
	"yacht": water_prompt_list,
	"bicycle": road_prompt_list,
	"armchair": indoor_prompt_list,
	"lamp": indoor_prompt_list,
	"canoe": water_prompt_list
} 

import os
import torch
import re

import os
import re
import random
import torch

# def generate_subject_categories():
# 	# Categorize subjects into prompt categories
# 	categories = {
# 		"road": [
# 			"sedan", "suv", "horse", "jeep", "tractor", 
# 			"motorcycle", "bicycle"
# 		],
# 		"water": [
# 			"boat", "dolphin", "ship", 
# 			"submarine", "yacht", "canoe"
# 		],
# 		"indoor": [
# 			"sofa", "teddy", "chair", 
# 			"armchair", "lamp"
# 		]
# 	}
# 	return categories

# def generate_subject_combinations(num_subjects=1):
# 	# Get categorized subjects
# 	subject_categories = generate_subject_categories()
	
# 	# Generate triplets for each category
# 	subject_triplets = {
# 		"road": [
# 			# First set of road subject triplets
# 			random.sample(subject_categories["road"], num_subjects),
# 			# Second set of road subject triplets
# 			random.sample(subject_categories["road"], num_subjects),
# 			# Third set of road subject triplets
# 			# random.sample(subject_categories["road"], num_subjects)
# 		],
# 		"water": [
# 			# Water subject triplets
# 			random.sample(subject_categories["water"], num_subjects),
# 			# random.sample(subject_categories["water"], num_subjects),
# 			# random.sample(subject_categories["water"], num_subjects)
# 		],
# 		"indoor": [
# 			# Indoor subject triplets
# 			random.sample(subject_categories["indoor"], num_subjects),
# 			# random.sample(subject_categories["indoor"], num_subjects),
# 			# random.sample(subject_categories["indoor"], num_subjects) 
# 		]
# 	}
	
# 	return subject_triplets

# @torch.no_grad() 
# def generate_images():
# 	# Load the Stable Diffusion 3.5 Pipeline
# 	pipe = StableDiffusion3GeneralInference.from_pretrained(
# 		"stabilityai/stable-diffusion-3.5-medium", 
# 	).to(accelerator.device)  # Move to GPU

# 	# Base output directory
# 	base_output_dir = f"./condition_till_timestep" 
# 	os.makedirs(base_output_dir, exist_ok=True)


# 	# Prompt lists
# 	prompt_lists = {
# 		"road": road_prompt_list,
# 		"water": water_prompt_list,
# 		"indoor": indoor_prompt_list
# 	}

# 	latents = torch.load("latents.pt") 

# 	# Process each category of subject triplets
# 	with accelerator.split_between_processes(list(subject_triplets.items()), apply_padding=False) as category_triplets: 
# 		for category, triplets in category_triplets:
# 			for subjects in triplets:
# 				# Create subject-triplet-specific output directory
# 				subjects_ = ["_".join(subject.split()) for subject in subjects] 
# 				subjects__ = "__".join(subjects_)  

# 				# Get the prompt list for this category
# 				prompt_list = prompt_lists[category]

# 				# Process prompts in batches
# 				batch_size = 4     
# 				latents = latents[:1].repeat(batch_size, 1, 1, 1) 
# 				for i in range(0, len(prompt_list), batch_size):
# 					# Get current batch of prompts
# 					batch_prompts = prompt_list[i:i+batch_size]
# 					# assert len(batch_prompts) == batch_size 
# 					latents = latents[:len(batch_prompts)]
# 					# print(f"{batch_prompts = }") 
					
# 					# Replace PLACEHOLDER with all three subjects
# 					placeholder_text = "" 
# 					for subject in subjects: 
# 						placeholder_text = placeholder_text + subject + " and " 
# 					placeholder_text = placeholder_text[:-5]  
# 					formatted_prompts = [
# 						prompt.replace("PLACEHOLDER", placeholder_text)  
# 						for prompt in batch_prompts
# 					]
# 					print(f"{formatted_prompts = }")

# 					#####################################  

# 					tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="tokenizer")   
# 					input_ids = [tokenizer.encode(formatted_prompt, return_tensors="pt")[0] for formatted_prompt in formatted_prompts]  
# 					print(f"{input_ids = }")
# 					prompts_tokens_strs = [tokenizer.batch_decode(input_ids_) for input_ids_ in input_ids]   
# 					print(f"{prompts_tokens_strs = }")
# 					images = pipe(
# 						prompt=formatted_prompts, 
# 						latents=latents, 
# 					).images 
# 					# Save images
# 					prompts_filenames = [] 
# 					for j, (image, prompt) in enumerate(zip(images, formatted_prompts)):
# 						# Create unique filename from prompt
# 						prompt_filename = sanitize_filename(prompt)
# 						prompts_filenames.append(prompt_filename) 
# 						save_dir = os.path.join(base_output_dir, prompt_filename) 
# 						os.makedirs(save_dir, exist_ok=True)   
# 						counter = 0 
# 						while True: 
# 							img_name = str(counter).zfill(4) + ".jpg" 
# 							save_path = osp.join(save_dir, img_name)
# 							if osp.exists(save_path): 
# 								counter += 1
# 								continue 
# 							# Save image
# 							image.save(save_path)
# 							print(f"saved: {save_path}") 
# 							break 

# 					torch.cuda.empty_cache() 

# 					images = pipe(
# 						prompt=formatted_prompts, 
# 						latents=latents, 
# 						prompts_tokens_strs=prompts_tokens_strs, 
# 						timesteps_to_vis_attn=TIMESTEPS_TO_VIS_ATTN, 
# 						attn_vis_dir=base_output_dir, 
# 						images=images, 
# 						prompts_filenames=prompts_filenames, 
# 					) 

# 					# for condition_till_timestep in [1, 5, 10, 15, 20, 25, 27]:  
# 					#     for text_encoder_to_remove in [""]: 
# 					#         print(f"GPU {accelerator.process_index} :: {text_encoder_to_remove = }, {condition_till_timestep = }") 
# 					#         # for vis_attn_layer in VIS_ATTN_LAYERS: 
# 					#         # Generate images
# 					#         images = pipe(
# 					#             prompt=formatted_prompts, 
# 					#             latents=latents, 
# 					#             # text_encoders_to_remove=[text_encoder_to_remove], 
# 					#             condition_till_timestep=condition_till_timestep, 
# 					#         ).images  
# 					#         for j, (image, prompt) in enumerate(zip(images, formatted_prompts)):
# 					#             # Create unique filename from prompt
# 					#             prompt_filename = sanitize_filename(prompt)
# 					#             prompts_filenames.append(prompt_filename) 
# 					#             save_dir = os.path.join(base_output_dir, prompt_filename) 
# 					#             os.makedirs(save_dir, exist_ok=True)   
# 					#             # img_name = str(counter).zfill(4) + ".jpg" 
# 					#             img_name = f"{condition_till_timestep}.jpg"  
# 					#             save_path = osp.join(save_dir, img_name)
# 					#             # Save image
# 					#             image.save(save_path)
# 					#             print(f"saved: {save_path}") 


# 					#####################################  
						
# 		print("Image generation complete!")

# Execute the image generation
if __name__ == "__main__":
	accelerator = Accelerator() 
	set_seed(1510 + accelerator.process_index)  
	# road_prompt_list = random.sample(road_prompt_list, 2)  
	# water_prompt_list = random.sample(water_prompt_list, 2) 
	# indoor_prompt_list = random.sample(indoor_prompt_list, 2)   
	# road_prompt_list = ["A photo of PLACEHOLDER"] 
	# water_prompt_list = ["A photo of PLACEHOLDER"] 
	# indoor_prompt_list = ["A photo of PLACEHOLDER"] 
	# Get predefined subject triplets
	# subject_triplets = generate_subject_combinations(1)    
	# generate_images() 
	with torch.no_grad(): 
		if WHICH_RUN != "": 
			ckpt_path = osp.join(ROOT_CKPTS_DIR, WHICH_RUN, f"checkpoint-{WHICH_STEP}") 
			assert osp.exists(ckpt_path), f"{ckpt_path = }" 

			args_path = osp.join(osp.split(ckpt_path)[0], "args.pkl") 
			assert osp.exists(args_path), f"{args_path = }" 
			with open(args_path, "rb") as f: 
				args = pickle.load(f) 

			logging_dir = Path(args.output_dir, args.logging_dir)
			accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
			kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
			accelerator = Accelerator(
				mixed_precision=args.mixed_precision,
				project_config=accelerator_project_config,
				kwargs_handlers=[kwargs],
			)

			weight_dtype = torch.float32
			if accelerator.mixed_precision == "fp16":
				weight_dtype = torch.float16
			elif accelerator.mixed_precision == "bf16":
				weight_dtype = torch.bfloat16

			vae = AutoencoderKL.from_pretrained(
				"madebyollin/sdxl-vae-fp16-fix", 
				subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
				revision=args.revision,
				variant=args.variant,
			)

			# pipeline = StableDiffusionXLPipeline.from_pretrained(
			# 	args.pretrained_model_name_or_path,
			# 	vae=vae, 
			# 	revision=args.revision,
			# 	variant=args.variant,
			# 	torch_dtype=weight_dtype,
			# ) 

			pipeline = SDXLWithCALL.from_pretrained(
				args.pretrained_model_name_or_path,
				vae=vae,
				revision=args.revision,
				variant=args.variant,
				torch_dtype=weight_dtype,
			) 
			pipeline.vae = vae 

			print(f"{args.pretrained_model_name_or_path = }") 
			# tokenizer_three = T5TokenizerFast.from_pretrained(
			# 	args.pretrained_model_name_or_path,
			# 	subfolder="tokenizer_3",
			# 	revision=args.revision,
			# )
			# text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
			# 	args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
			# )
			# vae = AutoencoderKL.from_pretrained(
			# 	args.pretrained_model_name_or_path,
			# 	subfolder="vae",
			# 	revision=args.revision,
			# 	variant=args.variant,
			# )
			# transformer = SD3Transformer2DModel.from_pretrained(
			# 	args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
			# )
			tokenizer_one = pipeline.tokenizer 
			tokenizer_two = pipeline.tokenizer_2 
			text_encoder_one = pipeline.text_encoder 
			text_encoder_two = pipeline.text_encoder_2 
			torch.cuda.empty_cache() 
			vae = pipeline.vae 
			unet = pipeline.unet 

			special_tokens_str = [] 
			for i in range(MAX_BATCH_SIZE): 
				for j in range(MAX_SUBJECTS): 
					special_tokens_str.append(f"<special_token_{i}_{j}>") 
			num_added_tokens = tokenizer_one.add_tokens(special_tokens_str)  
			assert num_added_tokens == MAX_SUBJECTS * MAX_BATCH_SIZE  
			text_encoder_one.resize_token_embeddings(len(tokenizer_one))  
			special_tokens_ints_one = tokenizer_one.convert_tokens_to_ids(special_tokens_str)  

			num_added_tokens = tokenizer_two.add_tokens(special_tokens_str)  
			assert num_added_tokens == MAX_SUBJECTS * MAX_BATCH_SIZE  
			text_encoder_two.resize_token_embeddings(len(tokenizer_two))  
			special_tokens_ints_two = tokenizer_two.convert_tokens_to_ids(special_tokens_str)  

			# num_added_tokens = tokenizer_three.add_tokens(special_tokens_str)  
			# assert num_added_tokens == MAX_SUBJECTS * MAX_BATCH_SIZE  
			# text_encoder_three.resize_token_embeddings(len(tokenizer_three))  
			# special_tokens_ints_three = tokenizer_three.convert_tokens_to_ids(special_tokens_str)  

			# special_encoder_part1 = XYZThetaConditioningSD3Part1WBatchNorm().to(accelerator.device)  
			# special_encoder_part2_one = XYZThetaConditioningSD3Part2(768).to(accelerator.device)  
			# special_encoder_part2_two = XYZThetaConditioningSD3Part2(1280).to(accelerator.device)  
			# special_encoder_part2_three = XYZThetaConditioningSD3Part2(4096).to(accelerator.device)  
			special_encoder = GoodPoseEmbedding(768) 
			special_encoder_two = GoodPoseEmbedding(768) 

			vae.requires_grad_(False)
			text_encoder_one.requires_grad_(False)
			# text_encoder_two.requires_grad_(False)
			special_encoder.requires_grad_(False) 
			special_encoder_two.requires_grad_(False) 


			vae.to(accelerator.device, dtype=torch.float32)
			unet.to(accelerator.device, dtype=weight_dtype)
			text_encoder_one.to(accelerator.device, dtype=weight_dtype)
			text_encoder_two.to(accelerator.device, dtype=weight_dtype) 

			# text_encoder_two.to(accelerator.device, dtype=weight_dtype)
			# special_encoder_part1.to(accelerator.device, dtype=weight_dtype) 
			# special_encoder_part2_one.to(accelerator.device, dtype=weight_dtype) 
			# special_encoder_part2_two.to(accelerator.device, dtype=weight_dtype) 
			# special_encoder_part2_three.to(accelerator.device, dtype=weight_dtype) 
			special_encoder.to(accelerator.device, dtype=weight_dtype) 
			special_encoder_two.to(accelerator.device, dtype=weight_dtype) 

			def get_lora_config(rank, use_dora, target_modules):
				base_config = {
					"r": rank,
					"lora_alpha": rank,
					"init_lora_weights": "gaussian",
					"target_modules": target_modules,
				}
				if use_dora:
					if is_peft_version("<", "0.9.0"):
						raise ValueError(
							"You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
						)
					else:
						base_config["use_dora"] = True

				return LoraConfig(**base_config)

			# now we will add new LoRA weights to the attention layers
			unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
			unet_lora_config = get_lora_config(rank=args.rank, use_dora=args.use_dora, target_modules=unet_target_modules)
			unet.add_adapter(unet_lora_config)

			if args.train_text_encoder:
				text_lora_config = LoraConfig(
					r=args.rank,
					lora_alpha=args.rank,
					init_lora_weights="gaussian",
					target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
				)
				text_encoder_one.add_adapter(text_lora_config)
				# text_encoder_two.add_adapter(text_lora_config)

			def unwrap_model(model):
				model = accelerator.unwrap_model(model)
				model = model._orig_mod if is_compiled_module(model) else model
				return model


			def load_model_hook(models, input_dir):
				unet_ = None
				text_encoder_one_ = None
				text_encoder_two_ = None

				while len(models) > 0:
					model = models.pop()

					if isinstance(model, type(unwrap_model(unet))):
						unet_ = model
					elif isinstance(model, type(unwrap_model(text_encoder_one))):
						text_encoder_one_ = model
					elif isinstance(model, type(unwrap_model(text_encoder_two))):
						text_encoder_two_ = model
					elif isinstance(model, type(unwrap_model(special_encoder))):
						special_encoder_ = model
					else:
						raise ValueError(f"unexpected save model: {model.__class__}")

				lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

				special_encoder_path = osp.join(input_dir, "special_encoder.pt") 
				special_encoder_state_dict = torch.load(special_encoder_path) 
				special_encoder_.load_state_dict(special_encoder_state_dict) 

				unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
				unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
				incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
				if incompatible_keys is not None:
					# check only for unexpected keys
					unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
					if unexpected_keys:
						logger.warning(
							f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
							f" {unexpected_keys}. "
						)

				if args.train_text_encoder:
					# Do we need to call `scale_lora_layers()` here?
					_set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

					_set_state_dict_into_text_encoder(
						lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
					)

				# Make sure the trainable params are in float32. This is again needed since the base models
				# are in `weight_dtype`. More details:
				# https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
				if args.mixed_precision == "fp16":
					models = [unet_]
					if args.train_text_encoder:
						models.extend([text_encoder_one_, text_encoder_two_])
					# only upcast trainable parameters (LoRA) into fp32
					cast_training_params(models)

			(
				unet,
				special_encoder, 
			) = accelerator.prepare(
				unet, special_encoder, 
			)
			if args.train_text_encoder: 
				assert text_encoder_one is not None
				# assert text_encoder_two is not None
				# text_encoder_one = accelerator.prepare(text_encoder_one) 
				text_encoder_one, text_encoder_two = accelerator.prepare(text_encoder_one, text_encoder_two) 

			accelerator.register_load_state_pre_hook(load_model_hook)
			accelerator.load_state(ckpt_path) 
			print(f"loaded all the models successfully...")

		
			num_samples = NUM_SAMPLES 
			conditioning_kwargs = {} 
			# conditioning_kwargs["special_encoder_part1"] = special_encoder_part1 
			# conditioning_kwargs["special_encoder_part2_one"] = special_encoder_part2_one 
			# conditioning_kwargs["special_encoder_part2_two"] = special_encoder_part2_two 
			# conditioning_kwargs["special_encoder_part2_three"] = special_encoder_part2_three 
			conditioning_kwargs["special_encoder"] = special_encoder  
			conditioning_kwargs["num_samples"] = NUM_SAMPLES 
			conditioning_kwargs["special_tokens_ints_one"] = special_tokens_ints_one 
			conditioning_kwargs["special_tokens_ints_two"] = special_tokens_ints_two  

			patch_custom_attention(unwrap_model(pipeline.unet)) 

		else: 
			pipeline = StableDiffusionXLPipeline.from_pretrained(
				"stabilityai/stable-diffusion-3.5-medium"
			).to(accelerator.device) 
			conditioning_kwargs = {} 
			print(f"made the pipeline from scratch...")

		# online_inference(pipeline, "/ssd_scratch/vaibhav/sd35_ablations/condition_till_timestep", accelerator , conditioning_kwargs)  
		# online_inference(pipeline, f"/ssd_scratch/vaibhav/results_{WHICH_RUN}_{WHICH_STEP}/", accelerator , conditioning_kwargs)  

		online_inference(pipeline, f"./results_{WHICH_RUN}_{WHICH_STEP}_{WHAT}/", accelerator , conditioning_kwargs)   