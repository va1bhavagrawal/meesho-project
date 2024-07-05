import sys
import os

# Check that the last command-line argument is provided
if len(sys.argv) < 2:
    print("Please provide a prompt as the last command-line argument.")
    sys.exit(1)

# Get the prompt from the last command-line argument
prompt = sys.argv[-1].replace("_", " ")
print(f"received prompt: {prompt}")

# Set up the Stable Diffusion pipeline
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe = pipe.to("cuda")

# Create the output folder
output_folder = os.path.join("..", "training_data_vaibhav", "prior_imgs_" + prompt.replace(" ", "_"))
print(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Generate and save the images
for i in range(100):
    image = pipe(prompt).images[0]
    image.save(os.path.join(output_folder, f"{i:03d}.jpg"))
    print(f"Saved image {i:03d}.jpg")
