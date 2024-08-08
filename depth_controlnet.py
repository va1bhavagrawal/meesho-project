from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
import sys  
from diffusers.utils import load_image


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16, device=0)  

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, device=0) 

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

def controlnet_depth_pipeline(image, prompt, device):   
    # depth_estimator = pipeline('depth-estimation')  
    # image = depth_estimator(image)
    # image = [img["depth"] for img in image] 
    # image = np.array(image)
    # image = image[..., None]
    # image = np.concatenate([image, image, image], axis=-1)
    # # for idx, img in enumerate(image):  
    # #     img = Image.fromarray(img) 
    # #     img.save(f"{idx}.jpg") 
    # image_ = [] 
    # for img in image: 
    #     img = Image.fromarray(img) 
    #     image_.append(img) 
    # image = image_ 

    # controlnet = ControlNetModel.from_pretrained(
    #     "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    # ).to(device) 

    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    # ).to(device) 

    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # # Remove if you do not have xformers installed
    # # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # # for installation instructions
    # pipe.enable_xformers_memory_efficient_attention()

    # pipe.enable_model_cpu_offload()

    image = pipe(prompt, image, num_inference_steps=20).images 

    # image.save('./images/stormtrooper_depth_out.png')
    return image 


# def controlnet_depth_pipeline2(image, prompt, device): 
#     depth_estimator = pipeline('depth-estimation')

#     image = image[0] 
#     prompt = prompt[0] 

#     # image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")

#     image = depth_estimator(image)['depth']
#     image = np.array(image)
#     image = image[:, :, None]
#     image = np.concatenate([image, image, image], axis=2)
#     image = Image.fromarray(image)

#     controlnet = ControlNetModel.from_pretrained(
#         "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
#     )

#     pipe = StableDiffusionControlNetPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
#     )

#     pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

#     # Remove if you do not have xformers installed
#     # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
#     # for installation instructions
#     pipe.enable_xformers_memory_efficient_attention()

#     pipe.enable_model_cpu_offload()

#     # image = pipe("Stormtrooper's lecture", image, num_inference_steps=20).images[0]
#     image = pipe(prompt, image, num_inference_steps=20).images[0]

#     image.save('./images/stormtrooper_depth_out.png')


if __name__ == "__main__": 
    import os 
    import os.path as osp  
    import shutil 

    REF_IMGS_DIR = "aer_8objects_lowelev_final"  
    OUTPUT_DIR = "./aer_8objects_lowelev_controlnet/"
    BS = 36  

    prompts_file = "prompts/prompts_0508.txt" 
    with open(prompts_file, "r") as f: 
        prompts = f.readlines() 
        prompts = [str(prompt) for prompt in prompts] 
    # for subject_ in os.listdir(REF_IMGS_DIR)[6:9]:  
    for subject_ in ["lion"]:   
        subject = " ".join(subject_.split("_")) 
        for prompt_idx, prompt in enumerate(prompts):  
            prompt_ = "_".join(prompt.split()) 
            # prompt = "a photo of " + prompt 
            # prompt = prompt.replace(f"SUBJECT", subject) 
            subject_path = osp.join(REF_IMGS_DIR, subject_) 
            img_names = sorted(os.listdir(subject_path))  
            img_names = [img_name for img_name in img_names if img_name.find(f".jpg") != -1] 
            # angles = [img_name.split(f".jpg")[0] for img_name in img_names] 
            num_images = len(img_names) 
            img_paths = [osp.join(subject_path, img_name) for img_name in img_names] 
            for start_idx in range(0, len(img_paths), BS):  
                end_idx = min(len(img_paths), start_idx + BS)  
                img_paths_batch = img_paths[start_idx : end_idx] 
                imgs = [Image.open(img_path) for img_path in img_paths_batch] 
                # prompt = prompt + "a photo of " 
                prompts = [] 
                for idx in range(end_idx - start_idx):  
                    assert "SUBJECT" in prompt 
                    img_name = img_names[start_idx + idx] 
                    _, view, a, e, r, _ = img_name.split(f"__") 
                    a = float(a) 
                    e = float(e) 
                    _r = float(r) 
                    assert view in ["front", "side", "back"] 
                    if view != "side": 
                        prompts.append(prompt.replace("a SUBJECT", f"a photo of the {view} side of a {subject}"))  
                    else: 
                        prompts.append(prompt.replace("a SUBJECT", f"a photo of a {subject}"))   
                    print(f"{prompts[-1]}")

                # prompts = [prompt] * BS  
                depth_estimator = pipeline('depth-estimation', device="cuda:0")   
                # depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda:0")  
                # image = depth_estimator(image)
                print(f"calculating depth maps...")
                depths = depth_estimator(imgs) 
                depths = [depth["depth"] for depth in depths] 
                print(f"done calculating depth maps, now saving them...")
                # for idx, depth in enumerate(depths): 
                #     depth["depth"].save(f"depth_{idx}.jpg") 
                # print(f"saved all depth maps!")
                # depths = [depth["depth"] for depth in depths] 
                # depths_np = [np.array(depth).astype(np.uint8) for depth in depths] 
                # imgs_np = [np.array(img).astype(np.uint8) for img in imgs] 
                # # imgs_masks = [img == np.array([0, 0, 0]).astype(np.uint8) for img in imgs] 
                
                # depths_masked = [] 
                # # for img_mask, depth in zip(imgs_masks, depths_np): 
                # for img, depth in zip(imgs, depths):  
                #     depth_masked = depth.copy()  
                #     # depth_masked[img_mask] = 0 
                #     img_ = np.array(img).astype(np.uint8) 
                #     # print(f"{img_.shape = }")
                #     # where_result = np.where(img_ == np.array([0, 0, 0]).astype(np.uint8))  
                #     where_result = np.where((img_[:, :, 0] < 10) & (img_[:, :, 1] < 10) & (img_[:, :, 2] < 10)) 
                #     # print(f"{where_result = }") 
                #     # print(f"{len(where_result) = }")
                #     # print(img_) 
                #     depth_masked = np.array(depth_masked).astype(np.uint8) 
                #     depth_masked[where_result] = 0 
                #     depths_masked.append(Image.fromarray(depth_masked)) 

                # depths = depths_masked 
                # # for idx, depth in enumerate(depths): 
                #     depth.save(f"depth_{idx}.jpg") 
                # print(f"saved all depth maps!")

                # for idx, img in enumerate(imgs): 
                #     depth = depth_estimator(img) 
                #     depth = depth["depth"] 
                #     print(f"{depth = }")
                #     depth.save(f"depth_{idx}.jpg") 
                # sys.exit(0) 
                    
                output_imgs = controlnet_depth_pipeline(depths, prompts, 0)     
                # output_imgs = [controlnet_depth_pipeline2(imgs, prompts, 0)] 
                for idx, img in enumerate(output_imgs):  
                    inp_img_name = img_names[start_idx + idx] 
                    _, view, a, e, r, _ = inp_img_name.split("__") 
                    a = float(a) 
                    e = float(e) 
                    r = float(r) 
                    save_path = osp.join(OUTPUT_DIR, subject_)  
                    os.makedirs(save_path, exist_ok=True) 
                    # save_path = osp.join(save_path, str(angles[start_idx + idx]) + "___" + f"prompt{prompt_idx}.jpg")  
                    save_path = osp.join(save_path, f"{a}__{e}__{r}__{view}__prompt{prompt_idx}.jpg")   
                    img.save(save_path) 