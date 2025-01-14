import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset 
from torchvision import transforms 
import pickle 

import os 
import os.path as osp 

import numpy as np 
import cv2 
from PIL import Image 
import random 

IMG_DIM = 1024 


class PriorImagesDataset(Dataset): 
    def __init__(self, args):  
        super().__init__() 
        self.args = args 
        self.imgs_dir = args.prior_imgs_dir 
        self.data = [] 
        self.size = args.resolution 

        for img_name in os.listdir(self.imgs_dir): 
            img_path = osp.join(self.imgs_dir, img_name) 
            prompt = img_name.replace("__.jpg", "") 
            example = {} 
            example["img_path"] = img_path 
            example["prompt"] = prompt 
            self.data.append(example) 


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if args.center_crop else transforms.RandomCrop(self.size), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self): 
        return len(self.data)  


    def __getitem__(self, idx): 
        raise NotImplementedError("this must be used by a MixingDatasets wrapper!") 



class RenderedImagesDataset(Dataset): 
    def __init__(self, args, imgs_dir, include_special_tokens):  
        super().__init__() 
        self.args = args 
        self.data = [] 
        self.size = args.resolution 
        self.imgs_dir = imgs_dir 

        for subjects_comb in os.listdir(self.imgs_dir): 
            subjects_comb_dir = osp.join(self.imgs_dir, subjects_comb) 
            for img_name in os.listdir(subjects_comb_dir): 
                if img_name.find("jpg") == -1: 
                    continue 
                example = {} 
                example["subjects_data"] = [] 
                img_path = osp.join(subjects_comb_dir, img_name) 
                pkl_path = img_path.replace("jpg", "pkl") 
                assert osp.exists(pkl_path), f"{pkl_path = }" 
                with open(pkl_path, "rb") as f: 
                    pkl_data = pickle.load(f) 
                example["img_path"] = img_path 
                subjects_ = subjects_comb.split("__")
                subjects = [" ".join(subject_.split("_")) for subject_ in subjects_]  
                subjects_details = img_name.replace(".jpg", "").split("__")[:-1]   
                assert len(subjects_details) == len(subjects), f"{subjects_details = }, {subjects = }" 
                for subject_idx in range(len(subjects)): 
                    subject_details = subjects_details[subject_idx] 
                    x, y, z, a = subject_details.split("_") 
                    subject_data = {} 
                    subject_data["x"] = float(x) 
                    subject_data["y"] = float(y) 
                    subject_data["theta"] = float(a) 
                    subject_data["name"] = subjects[subject_idx]  
                    subject_data["bbox"] = torch.as_tensor(pkl_data[f"obj{subject_idx+1}"]["bbox"], dtype=torch.float32) / IMG_DIM  
                    example["subjects_data"].append(subject_data) 
                prompt = self.args.rendered_imgs_prompt 
                assert prompt.find("PLACEHOLDER") != -1 
                placeholder_text = "" 
                subject_idx = 0 
                for subject_data in example["subjects_data"][:-1]: 
                    if include_special_tokens: 
                        placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']} and "  
                    else: 
                        placeholder_text = placeholder_text + f"{subject_data['name']} and " 
                    subject_idx += 1 
                for subject_data in example["subjects_data"][-1:]: 
                    if include_special_tokens: 
                        placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']}"  
                    else: 
                        placeholder_text = placeholder_text + f"{subject_data['name']}" 
                    subject_idx += 1 
                placeholder_text = placeholder_text.strip() 
                prompt = prompt.replace("PLACEHOLDER", placeholder_text) 
                # if args.cache_prompt_embeds: 
                #     embeds_path = osp.join(subjects_comb_dir, f"{'_'.join(prompt.split())}.pth") 
                #     embeds_path = embeds_path.replace("____", "__") 
                #     assert osp.exists(embeds_path), f"{embeds_path = }" 
                #     embeds = torch.load(embeds_path, map_location=torch.device("cpu"))  
                #     prompt_embeds = embeds["prompt_embeds"] 
                #     pooled_prompt_embeds = embeds["pooled_prompt_embeds"] 
                #     example["prompt_embeds"] = prompt_embeds 
                #     example["pooled_prompt_embeds"] = pooled_prompt_embeds 

                example["prompt"] = prompt 
                self.data.append(example) 


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if args.center_crop else transforms.RandomCrop(self.size), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self): 
        return len(self.data)  


    def __getitem__(self, idx): 
        raise NotImplementedError("this must be used by a MixingDatasets wrapper!") 


class ControlNetImagesDataset(Dataset): 
    def __init__(self, args, imgs_dir, ref_imgs_dir, include_special_tokens):  
        super().__init__() 
        self.args = args 
        self.data = [] 
        self.size = args.resolution 
        self.controlnet_prompts = args.controlnet_prompts 
        self.imgs_dir = imgs_dir 
        self.ref_imgs_dir = ref_imgs_dir 

        for subjects_comb in os.listdir(self.imgs_dir): 
            subjects_comb_dir = osp.join(self.imgs_dir, subjects_comb) 
            for img_name in os.listdir(subjects_comb_dir): 
                if (img_name.find("jpg") == -1) or (img_name.find("DEBUG") != -1): 
                    continue 
                # if "DEBUG__" + img_name not in os.listdir(subjects_comb_dir): 
                #     # this image has been removed during cleanup of controlnet dataset  
                #     continue 
                img_path = osp.join(subjects_comb_dir, img_name) 
                pkl_name = "__".join(img_name.split("__")[:-2]) + "__.pkl" 
                pkl_path = osp.join(self.ref_imgs_dir, subjects_comb, pkl_name)  
                
                assert osp.exists(pkl_path), f"{pkl_path = }"
                with open(pkl_path, "rb") as f: 
                    pkl_data = pickle.load(f) 
                example = {} 
                # if args.cache_prompt_embeds: 
                #     embeds_path = img_path.replace(".jpg", "__latents.pth") 
                #     embeds_path = embeds_path.replace("____", "__") 
                #     embeds = torch.load(embeds_path, map_location=torch.device("cpu")) 
                #     prompt_embeds = embeds["prompt_embeds"] 
                #     pooled_prompt_embeds = embeds["pooled_prompt_embeds"] 
                #     example["prompt_embeds"] = prompt_embeds 
                #     example["pooled_prompt_embeds"] = pooled_prompt_embeds 

                example["subjects_data"] = [] 
                example["img_path"] = img_path 
                subjects_ = subjects_comb.split("__")
                subjects = [" ".join(subject_.split("_")) for subject_ in subjects_]  
                subjects_details = img_name.replace(".jpg", "").split("__")[:-2]   
                prompt_idx = int(img_name.replace(".jpg", "").split("__")[-2].replace("prompt", ""))  
                assert len(subjects_details) == len(subjects), f"{subjects_details = }, {subjects = }" 
                for subject_idx in range(len(subjects)): 
                    subject_details = subjects_details[subject_idx] 
                    x, y, z, a = subject_details.split("_") 
                    subject_data = {} 
                    subject_data["x"] = float(x) 
                    subject_data["y"] = float(y) 
                    subject_data["theta"] = float(a) 
                    subject_data["name"] = subjects[subject_idx]  
                    subject_data["bbox"] = torch.as_tensor(pkl_data[f"obj{subject_idx+1}"]["bbox"], dtype=torch.float32) / IMG_DIM 
                    example["subjects_data"].append(subject_data) 
                prompt = self.controlnet_prompts[prompt_idx] 
                assert prompt.find("PLACEHOLDER") != -1 
                placeholder_text = "" 
                subject_idx = 0 
                # for subject_data in example["subjects_data"][:-1]: 
                #     placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']} and "  
                #     subject_idx += 1 
                # for subject_data in example["subjects_data"][-1:]: 
                #     placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']}"  
                #     subject_idx += 1 

                for subject_data in example["subjects_data"][:-1]: 
                    if include_special_tokens: 
                        placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']} and "  
                    else: 
                        placeholder_text = placeholder_text + f"{subject_data['name']} and " 
                    subject_idx += 1 
                for subject_data in example["subjects_data"][-1:]: 
                    if include_special_tokens: 
                        placeholder_text = placeholder_text + f"<special_token_{subject_idx}> {subject_data['name']}"  
                    else: 
                        placeholder_text = placeholder_text + f"{subject_data['name']}" 

                placeholder_text = placeholder_text.strip() 
                prompt = prompt.replace("PLACEHOLDER", placeholder_text) 
                example["prompt"] = prompt 
                self.data.append(example) 


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if args.center_crop else transforms.RandomCrop(self.size), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self): 
        return len(self.data)  


    def __getitem__(self, idx): 
        raise NotImplementedError("this must be used by a MixingDatasets wrapper!") 


class MixingDatasets(Dataset): 
    def __init__(self, args, datasets, ratios):  
        ratios = np.array(ratios).astype(np.int32) 
        assert np.all(ratios >= 1) 
        self.data = [] 
        self.datasets = [] 
        for dataset, ratio in zip(datasets, ratios): 
            self.data = self.data + dataset.data * ratio  
            self.datasets = self.datasets + [dataset] * len(self.data) * ratio 

    def __len__(self): 
        return len(self.data) 


    def __getitem__(self, idx): 
        example = {}  
        for k, v in self.data[idx].items(): 
            example[k] = v 
        img = Image.open(self.data[idx]["img_path"]) 
        if not img.mode == "RGB": 
            img = img.convert("RGB") 
        example["img_t"] = self.datasets[idx].image_transforms(img) 
        return example  