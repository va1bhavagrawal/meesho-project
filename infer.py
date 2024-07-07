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

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import patch_pipe
import torch
import torch.nn as nn
import os
import os.path as osp 
from transformers import CLIPTextModel, CLIPTokenizer
from training_scripts.continuous_word_mlp import continuous_word_mlp

model_id = "stabilityai/stable-diffusion-2-1"

# cur_model = "nonrigid-run" 
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# patch_pipe(
#     pipe,
#     "ckpts/" + cur_model + "_sd.safetensors",
#     patch_text=True,
#     patch_ti=True,
#     patch_unet=True,
# )

subject = "shoe"  

checkpoints = [
    # "59_s6000",
    # "119_s12000", 
    # "179_s18000", 
    # "239_s24000", 
    "299_s30000",
]

for checkpoint in checkpoints:
    print(f"doing checkpoint {checkpoint}") 
    patch_pipe(
        pipe,
        osp.join(f"ckpts/{subject}/lora_weight_e{checkpoint}.safetensors"), 
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    continuous_word_model = continuous_word_mlp(input_size = 2, output_size = 1024)
    continuous_word_model.load_state_dict(torch.load(f"ckpts/{subject}/mlp{checkpoint}.pt"))
    continuous_word_model.eval()

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    cur_token = 'sks'

    img_list = []

    corresponding_emb = tokenizer(cur_token,
            padding="do_not_pad", \
            truncation=True, \
            max_length = tokenizer.model_max_length).input_ids[1]

    interpolation_gap = 12
    values = []
    for idx in range(interpolation_gap):
        value = idx / interpolation_gap
        values.append(value)
        print(f"doing value {value}")
        p = torch.Tensor([value])
        # 15 the pre-defined number to normalize the attributes between 0 to 0.5
        x = torch.Tensor([torch.sin(2 * torch.pi * p), torch.cos(2 * torch.pi * p)]).cuda()
        continuous_word_model = continuous_word_model.cuda()
        mlp_emb = continuous_word_model(torch.unsqueeze(x, dim=0)).squeeze(0)
        
        """Replacing the rare token embeddings with the outputs_{checkpoint} of the MLP"""  
        with torch.no_grad():
            pipe.text_encoder.get_input_embeddings().weight[corresponding_emb] = mlp_emb

        torch.manual_seed(50)
        prompt = 'a sks photo of a bnha shoe'  
        # image = pipe(prompt, negative_prompt="bnha, worst quality", num_inference_steps=50, guidance_scale=6).images[0]
        image = pipe(prompt, negative_prompt="", num_inference_steps=50, guidance_scale=6).images[0]
        img_list.append(image)


    os.makedirs(subject, exist_ok=True)
    import shutil 
    if osp.exists(f"{subject}/outputs_{checkpoint}"):
        shutil.rmtree(f"{subject}/outputs_{checkpoint}") 
    os.makedirs(f"{subject}/outputs_{checkpoint}", exist_ok=True)

    for i in range(len(img_list)):
        img_list[i].save(osp.join(subject, f"outputs_{checkpoint}", f"{str(i).zfill(3)}__{values[i]}_.jpg"))
