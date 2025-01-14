import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

# init_image = load_image(url).convert("RGB")
import PIL 
from PIL import Image 
import os 
import os.path as osp 

root_imgs_dir = "results_sdxl_two_se_1e-4_1e-3_50000_"
for prompt_filename in os.listdir(root_imgs_dir): 
    prompt_path = osp.join(root_imgs_dir, prompt_filename) 
    prompt_imgs = [] 
    img_names = sorted(os.listdir(prompt_path))  
    img_names = [img_name for img_name in img_names if osp.isdir(osp.join(prompt_path, img_name))]
    for img_name in img_names: 
        img_path = osp.join(prompt_path, img_name, "img.jpg") 
        init_img = Image.open(img_path) 
        prompt_imgs.append(init_img) 
    prompt = " ".join(prompt_filename.split("_")) 
    print(f"{prompt = }")
    prompts = [prompt] * len(prompt_imgs) 
    refined_images = pipe(prompt=prompts, negative_prompt="blurry background, out of focus, soft background, haze, low resolution, unclear details, noise, grainy", image=prompt_imgs).images
    for img_name, refined_img in zip(img_names, refined_images):  
        refined_img.save(osp.join(prompt_path, img_name, "refined_img_w_neg_prompt.jpg")) 