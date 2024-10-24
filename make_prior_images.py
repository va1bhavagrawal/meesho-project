import torch
from diffusers import StableDiffusionPipeline
import os
import os.path as osp 
import shutil 
from accelerate import Accelerator 

accelerator = Accelerator() 

# Load the Stable Diffusion v2.1 pipeline from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", 
    torch_dtype=torch.float16
)
pipe.vae.to(accelerator.device) 
pipe.unet.to(accelerator.device) 
pipe.text_encoder.to(accelerator.device) 

NUM_PROMPTS = 100 

subjects = [
    "shoe", 
    "sedan", 
    "ostrich", 
    "helicopter", 
    "sofa", 
    "teddy", 
]

# Prompt
prompt = "a photo of a PLACEHOLDER"

# Directory to save images
for subject in subjects: 
    output_dir = osp.join("prior_imgs", subject) 
    os.makedirs(output_dir, exist_ok=True)

    # Generate 100 images
    num_images = 100
    with accelerator.split_between_processes(list(range(100)), apply_padding=False) as img_ids: 
        for i in img_ids:  
            # Generate image
            image = pipe(prompt).images[0]
            
            # Save image to output directory
            image_path = os.path.join(output_dir, f"{str(i).zfill(3)}.jpg") 
            image.save(image_path)
            print(f"Saved {image_path}")

        print(f"Generated {num_images} images and saved to {output_dir}.")
