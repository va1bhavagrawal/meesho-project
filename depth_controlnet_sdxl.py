import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image

from transformers import pipeline 


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

if __name__ == "__main__": 

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()



    # prompt = "stormtrooper lecture, photorealistic"
    # image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    # depth_image = get_depth_map(image)

    # images = pipe(
    #     prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
    # ).images
    # images[0]

    # images[0].save(f"stormtrooper.png")


    import os 
    import os.path as osp  
    import shutil 

    REF_IMGS_DIR = "aer_8objects_views"  
    OUTPUT_DIR = "./aer_8objects_controlnet_sdxl/"
    BS = 9  

    prompts_file = "prompts/prompts_2007.txt" 
    with open(prompts_file, "r") as f: 
        prompts = f.readlines() 
        prompts = [str(prompt) for prompt in prompts] 
    # for subject_ in os.listdir(REF_IMGS_DIR)[6:9]:  
    # prompts = [prompts[0]]  
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
                    assert "SUBJECT" in prompt, f"{prompt = }" 
                    img_name = img_names[start_idx + idx] 
                    _, view, a, e, r, _ = img_name.split(f"__") 
                    a = float(a) 
                    e = float(e) 
                    r = float(r) 
                    assert view in ["front", "side", "back", "top"] 
                    # if view != "side": 
                    #     prompts.append(prompt.replace("a SUBJECT", f"a photo of the {view} side of a {subject}"))  
                    # else: 
                    #     prompts.append(prompt.replace("a SUBJECT", f"a photo of a {subject}"))  
                    prompts.append(prompt.replace(f"SUBJECT", subject)) 
                    print(f"{prompts[-1]}")

                depth_maps = get_depth_map(imgs) 
                depth_maps = [] 
                for idx, img in enumerate(imgs): 
                    depth_maps.append(get_depth_map(img)) 
                    depth_maps[-1].save(f"depth_{idx}.jpg")  
                depths = depth_maps 


                # # prompts = [prompt] * BS  
                # # depth_estimator = pipeline('depth-estimation')  
                # depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda:0")  
                # # image = depth_estimator(image)
                # print(f"calculating depth maps...")
                # depths = depth_estimator(imgs) 
                # print(f"done calculating depth maps, now saving them...")
                # # for idx, depth in enumerate(depths): 
                # #     depth["depth"].save(f"depth_{idx}.jpg") 
                # # print(f"saved all depth maps!")
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
                # for idx, depth in enumerate(depths): 
                #     depth.save(f"depth_{idx}.jpg") 

                # for depth_idx, depth_map in enumerate(depth_maps): 
                #     depth_map.save(f"{depth_idx}.jpg") 
                # sys.exit(0) 

                output_imgs = pipe(
                    prompts, image=depths, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
                ).images

                # prompts = [prompt] * BS  
                # output_imgs = controlnet_depth_pipeline(imgs, prompts, 0)     
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
                    save_path = osp.join(save_path, f"{a}__{e}__{r}__prompt{prompt_idx}.jpg")   
                    img.save(save_path) 