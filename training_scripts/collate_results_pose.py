import os 
import os.path as osp 
import cv2 
from utils import * 
import PIL 
from PIL import Image 

# base_results_dir = "/ssd_scratch/vaibhav/sd35_ablations/condition_till_timestep" 
base_results_dir = "results_sdxl_two_se_1e-4_1e-3_50000_" 
images = [] 

for prompt_filename in os.listdir(base_results_dir): 
    all_imgs = {} 
    prompt_path = osp.join(base_results_dir, prompt_filename)   
    orientations = os.listdir(prompt_path)  
    all_imgs[prompt_filename] = {} 
    for orientation in orientations: 
        img_path = osp.join(prompt_path, orientation, "img.jpg") 

        #######################  

        # if img.find("skip") == -1: 
        #     continue 
        # skip_layer = img.replace("skip_", "") 
        # skip_layer = skip_layer.replace(".jpg", "") 
        # skip_layer = int(skip_layer) 
        # img = Image.open(img_path) 
        # if img_path.find("27") == -1: 
        #     continue 
        # if img.find("clip1") != -1: 
        #     timestep = int(img.replace("clip1_", "").replace(".jpg", ""))   
        #     all_imgs[prompt_filename]["clip1"] = Image.open(img_path)  
        # elif img.find("clip2") != -1: 
        #     all_imgs[prompt_filename]["clip2"] = Image.open(img_path)  
        # elif img.find("t5") != -1: 
        #     all_imgs[prompt_filename]["t5"] = Image.open(img_path)  
        timestep = int(orientation.replace(".jpg", "")) 
        all_imgs[prompt_filename][timestep] = Image.open(img_path) 

        #######################  
    # org_img_path = osp.join(prompt_path, "0000.jpg") 
    # all_imgs[prompt_filename]["org_img"] = Image.open(org_img_path) 

    # img_rows = [] 
    # captions_rows = [] 
    # # n rows each of 4 images 
    # for i, prompt_filename in enumerate(list(all_imgs.keys())):  
    #     row_imgs = []  
    #     row_captions = [] 
    #     # row_imgs = [all_imgs[prompt_filename]["org_img"], all_imgs[prompt_filename]["clip1"], all_imgs[prompt_filename]["clip2"], all_imgs[prompt_filename]["t5"]]  
    #     imgs_prompt = all_imgs[prompt_filename] 
    #     row_imgs = [imgs_prompt[1], imgs_prompt[5], imgs_prompt[10], imgs_prompt[15], imgs_prompt[20], imgs_prompt[25], imgs_prompt[27]] 
    #     # row_captions = ["org_img", "w/o clip1", "w/o clip2", "w/o t5"]  
    #     row_captions = ["1", "5", "10", "15", "20", "25", "27"] 
    #     img_rows.append(row_imgs) 
    #     captions_rows.append(row_captions) 

    # output_img = create_image_with_captions(img_rows, captions_rows) 
    # output_img = create_image_with_captions([[output_img]], [[" ".join(prompt_filename.split("_"))]]) 
    # output_img.save(osp.join(base_results_dir, prompt_filename, f"collated.jpg"))  
    # print(f"done {prompt_filename}")
    for i, prompt_filename in enumerate(list(all_imgs.keys())):  
        keys = sorted(list(all_imgs[prompt_filename].keys()))  
        imgs = [all_imgs[prompt_filename][k] for k in keys] 
        create_gif(imgs, osp.join(base_results_dir, prompt_filename, f"collated.gif")) 