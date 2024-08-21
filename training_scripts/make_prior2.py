import argparse
import os
import os.path as osp 
from diffusers import StableDiffusionPipeline
import torch
import sys 

# NUM_CLASS_IMAGES = 100
BS = 8  
NUM_IMGS_PER_PROMPT = 16 

# Set up the argument parser
parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
parser.add_argument('--file_id', help='The file_id of the images')
parser.add_argument('--prompts_file', help='the prompts file')
# parser.add_argument('--prompt', help='The prompt for generating images')



# Parse the command-line arguments
args = parser.parse_args()


with open(args.prompts_file, "r") as f: 
    prompts = f.readlines()  

    
# print(f"{prompts = }")
# sys.exit(0) 

# Get the file_id and prompt from the parsed arguments
file_id = args.file_id.strip() 
# prompt = args.prompt 
print(f"received file_id: {file_id}")
# print(f"received prompt: {prompt}")

# prompts_batch = [prompt] * BS 

subject = " ".join(file_id.split("_")) 

subject_prompts = [] 
for prompt in prompts: 
    prompt = prompt.replace(f"a SUBJECT", f"a photo of a {subject}") 
    for _ in range(NUM_IMGS_PER_PROMPT): 
        subject_prompts.append(prompt) 

prompts = subject_prompts 

# Set up the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to("cuda")

# Create the output folder
# output_folder = os.path.join("..", "training_data_vaibhav", "prior_imgs_" + file_id.strip().replace(" ", "_")) 
output_folder = osp.join(f"better_prior", f"{args.file_id}") 
print(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Generate and save the images
n_imgs = 0
# for i in range(() // BS):
total_n_generations = (NUM_IMGS_PER_PROMPT * len(prompts))  
for start_idx in range(0, total_n_generations, BS): 
    end_idx = min(start_idx + BS, total_n_generations) 
    prompts_batch = prompts[start_idx : end_idx]  
    for prompt in prompts_batch: 
        print(f"{prompt}")
    with torch.no_grad(): 
        images = pipe(prompts_batch).images
    for prompt, image in zip(prompts_batch, images): 
        save_dir = osp.join(output_folder, "_".join(prompt.split())) 
        os.makedirs(save_dir, exist_ok=True) 
        n_prompt_imgs = len(os.listdir(save_dir)) 
        # image.save(osp.join(output_folder, str(n_imgs).zfill(3) + ".jpg"))
        image.save(osp.join(save_dir, f"{str(n_prompt_imgs + 1).zfill(3)}.jpg")) 
        n_imgs += 1
    # print(f"saved images till {n_imgs}")
# import time 
# time.sleep(10000) 
