import argparse
import os
import os.path as osp 
from diffusers import StableDiffusionPipeline
import torch
import sys 

NUM_CLASS_IMAGES = 100
BS = 4

# Set up the argument parser
parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
parser.add_argument('--subject', help='The subject of the images')
parser.add_argument('--prompt', help='The prompt for generating images')

# Parse the command-line arguments
args = parser.parse_args()

# Get the subject and prompt from the parsed arguments
subject = args.subject
prompt = args.prompt 
print(f"received subject: {subject}")
print(f"received prompt: {prompt}")

prompts_batch = [prompt] * BS 

# Set up the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to("cuda")

# Create the output folder
output_folder = os.path.join("..", "training_data_vaibhav", "prior_imgs_" + subject.strip().replace(" ", "_")) 
print(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Generate and save the images
n_imgs = 0
for i in range(NUM_CLASS_IMAGES // BS):
    images = pipe(prompts_batch).images
    for image in images: 
        image.save(osp.join(output_folder, str(n_imgs).zfill(3) + ".jpg"))
        n_imgs += 1
    print(f"saved images till {n_imgs}")