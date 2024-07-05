import argparse
import os
from diffusers import StableDiffusionPipeline
import torch

# Set up the argument parser
parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
parser.add_argument('subject', required=True, help='The subject of the images')
parser.add_argument('prompt', required=True, help='The prompt for generating images')

# Parse the command-line arguments
args = parser.parse_args()

# Get the subject and prompt from the parsed arguments
subject = args.subject
prompt = args.prompt 
print(f"received prompt: {prompt}")

# Set up the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to("cuda")

# Create the output folder
output_folder = os.path.join("..", "training_data_vaibhav", "prior_imgs_" + "_".join(subject.split())) 
print(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Generate and save the images
for i in range(100):
    image = pipe(prompt).images[0]
    image.save(os.path.join(output_folder, f"{i:03d}.jpg"))
    print(f"Saved image {i:03d}.jpg")