import cv2
from PIL import Image
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import torch
import numpy as np
from diffusers.utils import load_image
import os 
import os.path as osp  
import shutil 
import sys 
import random 
from accelerate import Accelerator 
import math 
import matplotlib.pyplot as plt 

def find_view(a): 
    if (a >= 3*math.pi/8 and a <= 5*math.pi/8):  
        return "back" 
    if (a>=11*math.pi/8 and a <= 13*math.pi/8):  
        return "front" 
    return None 


CONDITIONING_SCALE = 0.5    
NUM_INFERENCE_STEPS = 20 
REF_IMGS_DIR = osp.join("2obj_renders_binned")  
OUTPUT_DIR = f"2obj_canny_{CONDITIONING_SCALE}_{NUM_INFERENCE_STEPS}/" 
DEBUG_DIR = f"2obj_canny_{CONDITIONING_SCALE}_{NUM_INFERENCE_STEPS}__debug" 
BS = 6   
USE_VIEW_HINTS = True 

def create_image_with_captions(image_rows, captions):
    # Check if the number of rows matches the number of caption rows
    if len(image_rows) != len(captions):
        raise ValueError("The number of caption rows must match the number of image rows.")

    # Load images and check if they are valid
    images_resized_rows = []
    for row in image_rows:
        images = [np.array(image).astype(np.uint8) for image in row]
        # print(f"{[image.shape for image in images]}") 
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded. Check the paths.")
        
        # Resize images to the same height for proper concatenation
        height = min(img.shape[0] for img in images)
        images_resized = [cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height)) for img in images]
        images_resized_rows.append(images_resized)

    # Create a blank image for captions
    caption_height = 50  # Height for the caption area
    caption_images = []

    for caption_row in captions:
        caption_images_row = []
        for caption in caption_row:
            # Create a blank image for the caption
            caption_img = np.ones((caption_height, images_resized_rows[0][0].shape[1], 3), dtype=np.uint8) * 255
            # Put the caption text on the blank image
            cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            caption_images_row.append(caption_img)
        caption_images.append(caption_images_row)

    # Concatenate images and captions for each row
    final_rows = []
    for images_resized, caption_images_row in zip(images_resized_rows, caption_images):
        # Concatenate images in the current row
        final_images = [cv2.vconcat([img, caption]) for img, caption in zip(images_resized, caption_images_row)]
        final_row = cv2.hconcat(final_images)
        final_rows.append(final_row)

    # Concatenate all rows vertically
    final_image = cv2.vconcat(final_rows)

    return Image.fromarray(final_image)


if __name__ == "__main__": 

    # controlnet = ControlNetModel.from_pretrained(
    #     "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, device="cuda:0", 
    # )


    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16, 
    )  
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16) 
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16, 
    )  

    # pipe.enable_model_cpu_offload()

    accelerator = Accelerator() 
    controlnet = controlnet.to(accelerator.device) 
    vae = vae.to(accelerator.device) 
    pipe = pipe.to(accelerator.device) 
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, device="cuda:0", 
    # )

    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    # pipe.enable_xformers_memory_efficient_attention()


    # image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/bird.png")
    # image = np.array(image)

    # LOGIC FOR ITERATING OVER THE IMAGES HERE!


    prompts_file = "prompts/prompts_2208.txt" 
    with open(prompts_file, "r") as f: 
        template_prompts = f.readlines() 
        template_prompts = [str(prompt).strip() for prompt in template_prompts] 

    # num_prompts_per_process = len(all_prompts) // accelerator.num_processes 

    for subject_pair_ in os.listdir(REF_IMGS_DIR):  
    # for subject_ in [sys.argv[-1].strip()]:   
        subject1_, subject2_ = subject_pair_.split("__") 
        subject1 = " ".join(subject1_.split("_")) 
        subject2 = " ".join(subject2_.split("_"))  

        os.makedirs(osp.join(OUTPUT_DIR, subject_pair_), exist_ok=True) 

        # prompt_ids = range(num_prompts_per_process * accelerator.process_index, (num_prompts_per_process + 1) * accelerator.process_index, 1)     
        # prompt_ids = range(accelerator.process_index * 3, (accelerator.process_index + 1) * 3, 1)   
        # prompt_ids = range(0, 12, 1)   
        proc_idx = accelerator.process_index 
        num_proc = accelerator.num_processes 
        # for prompt_idx, prompt in enumerate(prompts[10:14]):   
        # for prompt_idx in prompt_ids: 
        with accelerator.split_between_processes(list(range(8)), apply_padding=False) as prompt_ids: 
            print(f"{accelerator.process_index} is assigned {subject_pair_} :: {prompt_ids}") 
            for prompt_idx in prompt_ids: 
                print(f"{accelerator.process_index} is doing {subject_pair_} :: {prompt_idx}") 
                template_prompt = template_prompts[prompt_idx] 
                template_prompt_ = "_".join(template_prompt.split()) 
                # prompt = "a photo of " + prompt 
                # prompt = prompt.replace(f"SUBJECT", subject) 
                subject_pair_path = osp.join(REF_IMGS_DIR, subject_pair_) 
                img_names = sorted(os.listdir(subject_pair_path))  
                img_names = [img_name for img_name in img_names if img_name.find(f".jpg") != -1] 
                img_names = random.sample(img_names, 15)  
                # num_images = len(img_names) 
                img_paths = [osp.join(subject_pair_path, img_name) for img_name in img_names] 
                for img_path in img_paths: 
                    assert osp.exists(img_path) 

                for start_idx in range(0, len(img_paths), BS):  
                    torch.cuda.empty_cache() 
                    end_idx = min(len(img_paths), start_idx + BS)  
                    img_paths_batch = img_paths[start_idx : end_idx] 
                    imgs = [] 
                    for img_path in img_paths_batch: 
                        assert osp.exists(img_path) 
                    imgs = [Image.open(img_path) for img_path in img_paths_batch] 
                    # imgs = [cv2.imread(img_path) for img_path in img_paths_batch] 
                    # imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs] 
                    # imgs = [Image.fromarray(img) for img in imgs] 
                    # prompt = prompt + "a photo of " 
                    prompts = [] 
                    # for idx in range(end_idx - start_idx):  
                    #     assert "SUBJECT" in prompt 
                    #     img_name = img_names[start_idx + idx] 
                    #     _, view, a, e, r, _ = img_name.split(f"__") 
                    #     a = float(a) 
                    #     e = float(e) 
                    #     _r = float(r) 
                    #     assert view in ["front", "side", "back"] 
                    #     if view != "side": 
                    #         prompts.append(prompt.replace("a SUBJECT", f"a photo of the {view} side of a {subject}"))  
                    #     else: 
                    #         prompts.append(prompt.replace("a SUBJECT", f"a photo of a {subject}"))   
                    #     print(f"{prompts[-1]}")

                    for idx in range(end_idx - start_idx):  
                        assert "SUBJECT" in template_prompt, f"{template_prompt = }" 
                        if not USE_VIEW_HINTS: 
                            prompt = template_prompt.replace(f"a SUBJECT", f"a photo of a {subject1} and a {subject2}") 
                            prompts.append(prompt.strip()) 
                            continue 

                        img_name = img_names[start_idx + idx] 
                        subject1_coords, subject2_coords, _ = img_name.split("__") 
                        x1, y1, z1, a1 = subject1_coords.split("_") 
                        x2, y2, z2, a2 = subject2_coords.split("_") 
                        x1 = float(x1) 
                        y1 = float(y1) 
                        z1 = float(z1) 
                        a1 = float(a1) 
                        x2 = float(x2) 
                        y2 = float(y2) 
                        z2 = float(z2) 
                        a2 = float(a2) 

                        prompt = template_prompt.replace(f"a SUBJECT", "a photo of a SUBJECT1 and a SUBJECT2").strip()  

                        view1_hint = find_view(a1)  
                        view2_hint = find_view(a2) 

                        if view1_hint: 
                            prompt = prompt.replace(f"a SUBJECT1", f"the {view1_hint} view of a {subject1}") 
                        else: 
                            prompt = prompt.replace(f"SUBJECT1", f"{subject1}") 
                        if view2_hint: 
                            prompt = prompt.replace(f"a SUBJECT2", f"the {view2_hint} view of a {subject2}") 
                        else: 
                            prompt = prompt.replace(f"SUBJECT2", f"{subject2}") 
                        
                        prompts.append(prompt.strip()) 
                        
                        # _, view, a, e, r, _ = img_name.split(f"__") 
                        # a = float(a) 
                        # e = float(e) 
                        # _r = float(r) 
                        # assert view in ["front", "side", "back"] 
                        # if view != "side": 
                            # prompts.append(prompt.replace("a SUBJECT", f"a photo of the {view} side of a {subject}"))  
                        # else: 
                            # prompts.append(prompt.replace("a SUBJECT", f"a photo of a {subject}"))   

                        # prompts.append(prompt.replace("a SUBJECT", f"a realistic photo of a {subject}").strip())  
                        # print(f"{prompts[-1]}")

                    # prompts = [prompt] * BS  
                    # depth_estimator = pipeline('depth-estimation', device="cuda:0")   
                    # # depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda:0")  
                    # # image = depth_estimator(image)
                    # print(f"calculating depth maps...")
                    # depths = depth_estimator(imgs) 
                    # depths = [depth["depth"] for depth in depths] 
                    # print(f"done calculating depth maps, now saving them...")
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
                        
                    # output_imgs = controlnet_depth_pipeline(depths, prompts, 0)     
                    # output_imgs = [controlnet_depth_pipeline2(imgs, prompts, 0)] 


                    low_threshold = 100
                    high_threshold = 200

                    # image = cv2.Canny(image, low_threshold, high_threshold) 
                    canny_imgs = [cv2.Canny(np.array(img).astype(np.uint8), low_threshold, high_threshold) for img in imgs] 
                    canny_imgs = [img[:, :, None] for img in canny_imgs] 
                    canny_imgs = [np.repeat(img, 3, axis=-1) for img in canny_imgs] 
                    canny_imgs = [Image.fromarray(img) for img in canny_imgs] 
                    output_imgs = pipe(prompts, negative_prompt=["low quality, bad quality, sketches, unrealistic"] * len(prompts), image=canny_imgs, controlnet_conditioning_scale=CONDITIONING_SCALE).images  

                    for idx, img in enumerate(output_imgs):  
                        img_name = img_names[start_idx + idx] 
                        subject1_coords, subject2_coords, _ = img_name.split("__") 
                        x1, y1, z1, a1 = subject1_coords.split("_") 
                        x2, y2, z2, a2 = subject2_coords.split("_") 
                        x1 = float(x1) 
                        y1 = float(y1) 
                        z1 = float(z1) 
                        a1 = float(a1) 
                        x2 = float(x2) 
                        y2 = float(y2) 
                        z2 = float(z2) 
                        a2 = float(a2) 
                        save_path = osp.join(OUTPUT_DIR, subject_pair_)  
                        os.makedirs(save_path, exist_ok=True) 
                        # save_path = osp.join(save_path, str(angles[start_idx + idx]) + "___" + f"prompt{prompt_idx}.jpg")  
                        save_path = osp.join(save_path, f"{img_name.replace(f'.jpg', '').strip()}__prompt{prompt_idx}__.jpg")     
                        img.save(save_path) 

                        save_path_debug = osp.join(DEBUG_DIR, subject_pair_) 
                        os.makedirs(save_path_debug, exist_ok=True) 
                        save_path_debug = osp.join(save_path_debug, f"{img_name.replace(f'.jpg', '').strip()}__prompt{prompt_idx}__.jpg")  
                        # plt.figure(figsize=(60, 10)) 
                        # fig, axs = plt.subplots(1, 3) 
                        # axs[0].imshow(img) 
                        ref_img = Image.open(osp.join(REF_IMGS_DIR, subject_pair_, img_name))
                        # axs[1].imshow(ref_img)  
                        # axs[2].imshow(canny_imgs[idx])  
                        # fig.suptitle(prompts[prompt_idx], fontsize=3) 
                        # plt.savefig(save_path_debug) 
                        image_rows = [[img, ref_img.convert("RGB"), canny_imgs[idx]]]  
                        caption_rows = [["ControlNet", "ref. img", "Canny"]] 
                        concat_img = create_image_with_captions(image_rows, caption_rows) 
                        concat_img = create_image_with_captions([[concat_img]], [[prompts[idx]]]) 
                        concat_img.save(save_path_debug) 

                    torch.cuda.empty_cache() 