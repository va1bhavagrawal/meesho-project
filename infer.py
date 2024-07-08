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
import os
import os.path as osp 

ROOT_CKPT_DIR = "."

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import patch_pipe
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from training_scripts.continuous_word_mlp import continuous_word_mlp
import matplotlib.pyplot as plt 
import shutil 

model_id = "stabilityai/stable-diffusion-2-1"

# cur_model = "nonrigid-run" 
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

file_id = "template_truck"  

checkpoints = [
    "166_s30000",
]

def generate_prompts(subject="bnha pickup truck", use_sks=True, prompts_file="prompts/prompts_new.txt"):
    prompts_file = open(prompts_file, "r") 
    prompts = []
    for line in prompts_file.readlines():
        if len(line) < 2: # empty line 
            break
        prompt = str(line)
        prompt = "a" + prompt[1:]
        if use_sks:
            prompt = "a sks photo of " + prompt 
        prompt = prompt.replace(f"pickup truck", subject)
        prompts.append(prompt)
    return prompts


subjects = [
    "bnha pickup truck",
    "pickup truck",
    "sedan car",
    "sporty car",
    "motorbike",
]

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
os.makedirs(file_id, exist_ok=True)
for checkpoint in checkpoints:
    if osp.exists(f"{file_id}/outputs_{checkpoint}"):
        shutil.rmtree(f"{file_id}/outputs_{checkpoint}") 
    os.makedirs(f"{file_id}/outputs_{checkpoint}", exist_ok=True)
    patch_pipe(
        pipe,
        osp.join(ROOT_CKPT_DIR, f"ckpts/{file_id}/lora_weight_e{checkpoint}.safetensors"), 
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    for subject in subjects:
        prompts = generate_prompts(subject, use_sks=True)
        for prompt in prompts:
            print(f"doing prompt: {prompt}")
            prompt_ = "_".join(prompt.split()) 
            os.makedirs(f"{file_id}/outputs_{checkpoint}/{prompt_}", exist_ok=True)
            continuous_word_model = continuous_word_mlp(input_size = 2, output_size = 1024)
            new_state_dict = {}
            state_dict = torch.load(osp.join(ROOT_CKPT_DIR, f"ckpts/{file_id}/mlp{checkpoint}.pt"))
            for key, value in state_dict.items():
                if key.find(f"module") != -1:
                    new_state_dict[key.replace(f"module.", "")] = value
                else:
                    new_state_dict[key] = value  
            continuous_word_model.load_state_dict(new_state_dict) 
            continuous_word_model.eval()

            img_list = []
            cur_token = 'sks'
            corresponding_emb = tokenizer(cur_token,
                    padding="do_not_pad", \
                    truncation=True, \
                    max_length = tokenizer.model_max_length).input_ids[1]

            interpolation_gap = 12
            values = []
            for idx in range(interpolation_gap):
                value = idx / interpolation_gap
                values.append(value)
                p = torch.Tensor([value])
                x = torch.Tensor([torch.sin(2 * torch.pi * p), torch.cos(2 * torch.pi * p)]).cuda()
                continuous_word_model = continuous_word_model.cuda()
                mlp_emb = continuous_word_model(torch.unsqueeze(x, dim=0)).squeeze(0)
                
                """Replacing the rare token embeddings with the outputs_{checkpoint} of the MLP"""  
                with torch.no_grad():
                    pipe.text_encoder.get_input_embeddings().weight[corresponding_emb] = mlp_emb

                torch.manual_seed(50)
                # image = pipe(prompt, negative_prompt="bnha, worst quality", num_inference_steps=50, guidance_scale=6).images[0]
                image = pipe(prompt, negative_prompt="worst quality", num_inference_steps=50, guidance_scale=6).images[0]
                img_list.append(image)
                plt.imshow(image)
                plt.savefig(f"{file_id}/outputs_{checkpoint}/{prompt_}/{str(idx).zfill(3)}__{values[idx]}_.jpg") 