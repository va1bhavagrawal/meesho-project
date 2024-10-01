import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Load the Stable Diffusion 2.1 model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# print(pipe.unet)
# import sys 
# sys.exit(0) 

# Change the scheduler to DPMSolverMultistepScheduler for better performance
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move the pipeline to GPU for faster inference
pipe = pipe.to("cuda")

# Define your prompt
prompt = "a photo of an astronaut riding a horse on mars"

# Generate the image
image = pipe(prompt).images[0]

# Save the generated image
image.save("astronaut_rides_horse.png")