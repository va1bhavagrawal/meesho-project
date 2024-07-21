import diffusers 

pipe = diffusers.StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1") 
tokenizer = pipe.tokenizer 
prompt = "truck" 
input_ids = tokenizer(prompt).input_ids 
print(f"{input_ids = }")