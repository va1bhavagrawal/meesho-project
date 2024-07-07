from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
import os 
import os.path as osp 
import matplotlib.pyplot as plt 
import random 

inp_imgs_path = f"ref_imgs_template_truck/"
depth_estimator = pipeline('depth-estimation')


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# prompts = [
#     "A pickup truck navigating a winding mountain road, with snow-capped peaks, evergreen forests, and a frozen lake in the background. The sky is a crisp, clear blue.",
# ]
prompts = []
prompts_file = open("prompts_blue_truck.txt", "r")
for line in prompts_file.readlines():
    prompts.append(str(line))

for prompt in prompts:
    n_words = len(prompt.split())
    assert n_words < 77

print(f"sanity checks on prompts passed!")

all_files = os.listdir(inp_imgs_path)
img_files = [file for file in all_files if file.find(".png") != -1 or file.find(".jpg") != -1]
os.makedirs(f"images6", exist_ok=True)
for img_idx, img_file in enumerate(img_files):
    count = 10 
    while count > 0:
        prompt_idx = random.randint(0, len(prompts) - 1)
        prompt = prompts[prompt_idx]
        img_path = osp.join(inp_imgs_path, img_file)
        print(f"{img_path = }")
        image = Image.open(img_path)
        image = depth_estimator(image)['depth']
        plt.imshow(image)
        plt.savefig(f"depth_{img_idx}.jpg")
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        image = pipe(prompt, image, num_inference_steps=20).images[0]
        image.save(osp.join(f"images6", img_file.replace(".jpg", "") + "__" + f"prompt{str(prompt_idx).zfill(3)}" + ".jpg"))
        count -= 1