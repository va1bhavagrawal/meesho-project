import diffusers 
import sys 

pipe = diffusers.StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1") 
tokenizer = pipe.tokenizer 
prompts = [
    "pickup" 
] 
all_input_ids = tokenizer(prompts).input_ids 
for input_ids, prompt in zip(all_input_ids, prompts): 
    print(f"{input_ids = }, {prompt = }")