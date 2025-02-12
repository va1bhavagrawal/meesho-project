import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path
from typing import Optional
import random
import re
from PIL import Image 
import glob

import os 
import os.path as osp 

from infer_online import UNIQUE_TOKENS 

import pickle 

MAX_SUBJECTS_PER_EXAMPLE = 2  
RAW_IMG_SIZE = 1024 

# class PromptDataset(Dataset):
#     "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

#     def __init__(self, num_samples=24, subjects=None):
#         self.num_samples = num_samples 
#         self.subjects = subjects 
#         assert self.subjects is not None 
#         # if self.subjects is None: 
#         #     self.subjects = [
#         #         "bnha pickup truck",
#         #         "bnha motorbike",  
#         #         "bnha horse", 
#         #         "bnha lion", 
#         #     ] 
#         # self.subjects2 = [
#         #     "bicycle", 
#         #     "tractor", 
#         #     "sports car", 
#         #     "brad pitt", 
#         # ]

#         self.template_prompts = [
#             # prompts testing if the model can follow the prompt to create an 'environment'
#             "a photo of a SUBJECT on a remote country road, surrounded by rolling hills, vast open fields and tall trees", 
#             "a photo of a SUBJECT on a bustling city street, surrounded by towering skyscrapers and neon lights",
#             "a photo of a SUBJECT in front of a dark background",  
#             "a photo of a SUBJECT beside a field of blooming sunflowers, with snowy mountain ranges in the distance.",  
#             "a SUBJECT on a tropical beach, with palm trees swaying and waves crashing on the shore", 
#             "a SUBJECT in a colorful tulip field, with windmills in the background", 
#         ]
#         # this is just an indicator of azimuth, not the exact value 
#         self.azimuths = torch.arange(num_samples)  
#         self.prompt_wise_subjects, self.prompts = self.generate_prompts()
#         assert len(self.prompt_wise_subjects) == len(self.prompts) 


#     def generate_prompts(self):  
#         prompts = []
#         prompt_wise_subjects = [] 
#         for subject in self.subjects: 
#             for prompt in self.template_prompts:
#                 # if use_sks:
#                 #     prompt = "a sks photo of " + prompt 
#                 # else: 
#                 #     prompt = "a photo of " + prompt 
#                 subject_without_bnha = subject.replace("bnha", "").strip()  
#                 subject_without_bnha = " " + subject_without_bnha + " " 
#                 prompt_ = prompt.replace(f"SUBJECT", "bnha") 
#                 # we DO NOT want the subject to be present in the prompt text 
#                 assert prompt_.find(subject) == -1, f"{prompt_ = }, {prompt = }, {subject = }" 
#                 assert prompt_.find(subject_without_bnha) == -1, f"{prompt_ = }, {prompt = }, {subject_without_bnha = }, {subject = }" 
#                 prompts.append(prompt_)  
#                 prompt_wise_subjects.append(subject) 
#         return prompt_wise_subjects, prompts  


#     def __len__(self):
#         return len(self.subjects) * len(self.template_prompts) * self.num_samples


#     def __getitem__(self, index):
#         return self.data[index] 


class DisentangleDataset(Dataset): 
    def __init__(
        self,
        args, 
        tokenizer,
        ref_imgs_dirs, 
        controlnet_imgs_dirs, 
        num_steps, 
    ): 
        self.args = args 
        # controlnet prompts are provided as a list, not as a filepath.
        self.tokenizer = tokenizer 
        # self.ref_imgs_dir = ref_imgs_dir 
        self.num_steps = num_steps 

        img_transforms = []

        self.ref_imgs_dirs = ref_imgs_dirs  
        self.controlnet_imgs_dirs = controlnet_imgs_dirs 

        if args.resize:
            img_transforms.append(
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if args.center_crop:
            img_transforms.append(transforms.CenterCrop(args.resolution)) 
        if args.color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        # if args.h_flip:
        #     img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        if self.args.use_ref_images: 
            self.subjects_combs_ref = {}  
            for ref_imgs_dir in ref_imgs_dirs: 
                if not osp.exists(ref_imgs_dir): 
                    continue 
                self.subjects_combs_ref[ref_imgs_dir] = [] 
                subjects_combs_ = os.listdir(ref_imgs_dir) 
                for subjects_comb_ in subjects_combs_: 
                    self.subjects_combs_ref[ref_imgs_dir].append(subjects_comb_)  

        if self.args.use_controlnet_images: 
            self.subjects_combs_controlnet = {}  
            for controlnet_imgs_dir in controlnet_imgs_dirs: 
                if not osp.exists(controlnet_imgs_dir): 
                    continue 
                self.subjects_combs_controlnet[controlnet_imgs_dir] = [] 
                subjects_combs_ = os.listdir(controlnet_imgs_dir) 
                for subjects_comb_ in subjects_combs_: 
                    self.subjects_combs_controlnet[controlnet_imgs_dir].append(subjects_comb_)  


    def __len__(self):
        # return self.args.max_train_steps  
        # a window of size 16 is there to prevent border cases 
        return self.num_steps + 16   


    def __getitem__(self, index): 
        example = {} 

        # selecting the subject according to the index -- this is not randomized, every subject will get equal representation for sure  
        # subject = self.args.subjects[index % len(self.args.subjects)] 
        # subject_ = "_".join(subject.split()) 
        # subject_ref_dir = osp.join(self.args.instance_data_dir, subject_)
        # assert osp.exists(subject_ref_dir) 

        # example["subject"] = subject 
        # if index > self.args.stage1_steps or self.args.stage1_steps == -1:  
        #     subjects_combs_ = sorted(os.listdir(self.args.instance_data_dir))  
        #     subjects_comb_ = subjects_combs_[index % len(self.args.subjects_combs)] 
        #     subjects_ = subjects_comb_.split("__") 
        #     subjects = [" ".join(subject_.split("_")) for subject_ in subjects_] 
        #     example["subjects"] = subjects 
        # else:  
        #     subjects_combs_ = sorted(os.listdir(self.args.instance_data_dir_singlesub)) 
        #     subjects_comb_ = subjects_combs_[index % len(subjects_combs_)]   
        #     single_subject = " ".join(subjects_comb_.split("_")) 
        #     subjects = [single_subject] 
        #     example["subjects"] = subjects  

        # subjects_combs_ = sorted(os.listdir(self.ref_imgs_dir))  
        # # print(f"{subjects_combs_ = }")
        # subjects_comb_ = subjects_combs_[index % len(subjects_combs_)]   
        # # print(f"{subjects_comb_ = }") 
        # subjects_ = subjects_comb_.split("__") 
        # subjects = [" ".join(subject_.split("_")) for subject_ in subjects_] 
        # example["subjects"] = subjects 


        # selecting the random view for the chosen subject 
        # random_ref_img = random.choice(os.listdir(subject_ref_dir))  
        # angle = float(random_ref_img.split(f".jpg")[0])  
        # example["scaler"] = angle 

        # choosing from the instance images, not the augmentation 
        # if index % 5 != 0: 
        # only choosing the controlnet images in this one 
        # if False:  

        # deciding the subject combination must be deterministic and not random 
        if self.args.use_ref_images: 
            used_ref_imgs_dir = self.ref_imgs_dirs[index % len(self.ref_imgs_dirs)]   

        if self.args.use_controlnet_images: 
            used_controlnet_imgs_dir = self.controlnet_imgs_dirs[index % len(self.controlnet_imgs_dirs)]  

        subjects_combs_ = sorted(os.listdir(used_ref_imgs_dir))   
        # print(f"{subjects_combs_ = }")
        subjects_comb_ = subjects_combs_[index % len(subjects_combs_)]   
        # print(f"{subjects_comb_ = }") 
        subjects_ = subjects_comb_.split("__") 
        subjects = [" ".join(subject_.split("_")) for subject_ in subjects_] 
        example["subjects"] = subjects 

    
        unique_strings = []  
        for asset_idx in range(len(subjects)): 
            unique_string_subject = "" 
            for token_idx in range(self.args.merged_emb_dim // 1024): 
                unique_string_subject = unique_string_subject + f"{UNIQUE_TOKENS[f'{asset_idx}_{token_idx}']} " 
            unique_string_subject = unique_string_subject.strip() 
            unique_strings.append(unique_string_subject) 

        assert self.args.use_ref_images or self.args.use_controlnet_images 
        if not self.args.use_controlnet_images or (self.args.use_ref_images and index % 5 != 0): 
            example["controlnet"] = False 
            # if len(example["subjects"]) == 2: 
            #     subjects_comb_ref_dir = osp.join(self.args.instance_data_dir, subjects_comb_) 
            # elif len(example["subjects"]) == 1: 
            #     subjects_comb_ref_dir = osp.join(self.args.instance_data_dir_singlesub, subjects_comb_) 
            # else: 
            #     assert False 
            subjects_comb_ref_dir = osp.join(used_ref_imgs_dir, subjects_comb_) 
            imgs_list = os.listdir(subjects_comb_ref_dir) 
            imgs_list = [img_name for img_name in imgs_list if img_name.find("jpg") != -1 or img_name.find("png") != -1] 
            random_ref_img = random.choice(imgs_list)   
            img_path = osp.join(osp.join(subjects_comb_ref_dir, random_ref_img)) 
            pkl_path = img_path.replace("jpg", "pkl") 
            assert osp.exists(pkl_path), f"{pkl_path = }"
            with open(pkl_path, "rb") as f: 
                pkl_data = pickle.load(f) 
            all_2d_x = [] 
            all_2d_y = [] 
            all_bboxes = [] 
            assert len(pkl_data.keys()) == len(example["subjects"]) 
            for asset_idx in range(len(pkl_data.keys())): 
                bbox = pkl_data[f"obj{asset_idx+1}"]["bbox"] 
                all_2d_x.append((bbox[0] + bbox[2]) / (2 * RAW_IMG_SIZE))  
                all_2d_y.append((bbox[1] + bbox[3]) / (2 * RAW_IMG_SIZE))  
                all_bboxes.append(torch.tensor(bbox) / RAW_IMG_SIZE)  

            # a, e, r, x, y, _ = random_ref_img.split("__") 
            # a = float(a) 
            # e = float(e) 
            # r = float(r) 
            # x = int(x) 
            # y = int(y) 
            subjects_data = random_ref_img.split("__") 
            subjects_data = subjects_data[:-1] 
            all_x = []  
            all_y = [] 
            all_z = [] 
            all_a = [] 
            assert len(subjects_data) == len(example["subjects"])  
            assert len(subjects_data) <= MAX_SUBJECTS_PER_EXAMPLE 
            for asset_idx in range(len(example["subjects"])): 
                one_subject_data = subjects_data[asset_idx] 
                x, y, z, a = one_subject_data.split("_") 
                all_x.append(float(x)) 
                all_y.append(float(y)) 
                all_z.append(float(z)) 
                all_a.append(float(a)) 

            example["scalers"] = all_a   
            example["bboxes"] = all_bboxes 
            example["2d_xs"] = all_2d_x 
            example["2d_ys"] = all_2d_y  
            template_prompt = "a photo of PLACEHOLDER" 
            placeholder_text = "a SUBJECT0 "  
            for asset_idx in range(1, len(example["subjects"])):  
                placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} " 
            placeholder_text = placeholder_text.strip() 
            template_prompt = template_prompt.replace("PLACEHOLDER", placeholder_text) 
            if not self.args.include_class_in_prompt: 
                for asset_idx in range(len(example["subjects"])):    
                    assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                    template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]}") 
            else: 
                for asset_idx in range(len(subjects)): 
                    assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                    template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {subjects[asset_idx]}") 

            example["prompt"] = template_prompt   
            example["prompt_ids"] = self.tokenizer(
                example["prompt"], 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length, 
            ).input_ids 

            assert osp.exists(img_path)  
            img = Image.open(img_path)  

        # choosing from the controlnet augmentation 
        elif self.args.use_controlnet_images:  
            example["controlnet"] = True  
            # subject_angle_controlnet_dir = osp.join(self.args.controlnet_data_dir, subject_, str(angle))  

            # subject_controlnet_dir = osp.join(self.args.controlnet_data_dir, subject_) 
            # avlble_imgs = os.listdir(subject_controlnet_dir)  
            # chosen_img = random.choice(avlble_imgs) 
            # a, e, r, x, y, _ = chosen_img.split("__") 
            # a = float(a) 
            # e = float(e) 
            # r = float(r) 
            # x = int(x) 
            # y = int(y) 
            # example["scaler"] = a 

            # avlble_imgs = os.listdir(subject_angle_controlnet_dir) 
            # chosen_img = random.choice(avlble_imgs) 

            subjects_comb_controlnet_dir = osp.join(used_controlnet_imgs_dir, subjects_comb_) 
            imgs_list = os.listdir(subjects_comb_controlnet_dir)  
            imgs_list = [img_name for img_name in imgs_list if img_name.find("jpg") != -1 or img_name.find("png") != -1] 
            random_controlnet_img = random.choice(imgs_list)     
            img_path = osp.join(subjects_comb_controlnet_dir, random_controlnet_img) 
            subjects_data = random_controlnet_img.split("__") 
            subjects_data = subjects_data[:-1] 
            whichprompt = subjects_data[-1] 
            subjects_data = subjects_data[:-1] 
            pkl_path = osp.join(used_ref_imgs_dir, subjects_comb_, f"{'__'.join(subjects_data)}__.pkl") 
            assert osp.exists(pkl_path), f"{pkl_path = }" 
            with open(pkl_path, "rb") as f: 
                pkl_data = pickle.load(f) 
            all_2d_x = [] 
            all_2d_y = [] 
            all_bboxes = [] 
            assert len(pkl_data.keys()) == len(example["subjects"]) 
            for asset_idx in range(len(pkl_data.keys())): 
                bbox = pkl_data[f"obj{asset_idx+1}"]["bbox"] 
                all_2d_x.append((bbox[0] + bbox[2]) / (2 * RAW_IMG_SIZE))   
                all_2d_y.append((bbox[1] + bbox[3]) / (2 * RAW_IMG_SIZE))  
                all_bboxes.append(torch.tensor(bbox) / RAW_IMG_SIZE) 

            all_x = [] 
            all_y = [] 
            all_z = [] 
            all_a = [] 
            assert len(subjects_data) == len(example["subjects"]), f"{used_controlnet_imgs_dir = }, {used_ref_imgs_dir = }" 
            for asset_idx in range(len(example["subjects"])): 
                one_subject_data = subjects_data[asset_idx] 
                x, y, z, a = one_subject_data.split("_") 
                all_x.append(float(x))
                all_y.append(float(y)) 
                all_z.append(float(z)) 
                all_a.append(float(a)) 

            example["scalers"] = all_a 
            example["2d_xs"] = all_2d_x 
            example["2d_ys"] = all_2d_y 
            example["bboxes"] = all_bboxes 
            prompt_idx = int(whichprompt.replace("prompt", "").strip()) 
            template_prompt = self.args.controlnet_prompts[prompt_idx] 
            placeholder_text = "a SUBJECT0 " 
            for asset_idx in range(1, len(example["subjects"])):  
                placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} "  
            placeholder_text = placeholder_text.strip() 
            template_prompt = template_prompt.replace("PLACEHOLDER", placeholder_text) 
            if not self.args.include_class_in_prompt: 
                for asset_idx in range(len(example["subjects"])): 
                    assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                    template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{subjects[asset_idx]}")  
            else: 
                for asset_idx in range(len(subjects)): 
                    assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                    template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {subjects[asset_idx]}") 

            example["prompt"] = template_prompt 
            example["prompt_ids"] = self.tokenizer( 
                example["prompt"], 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length, 
            ).input_ids 

            assert osp.exists(img_path) 
            img = Image.open(img_path)   


        if not img.mode == "RGB":  
            img = img.convert("RGB") 
        example["img"] = self.image_transforms(img)  


        if self.args.with_prior_preservation: 
            random_subject_ = random.choice(subjects_)  
            random_subject = " ".join(random_subject_.split("_")) 
            example["prior_subject"] = random_subject 
            subject_class_imgs_path = osp.join(self.args.class_data_dir, random_subject_)   
            assert len(os.listdir(subject_class_imgs_path)) == 100, f"{len(os.listdir(subject_class_imgs_path)) = }"  
            n_class_images = len(os.listdir(subject_class_imgs_path)) 
            class_img_name = str(index % n_class_images).zfill(3) + ".jpg"  
            class_img_path = osp.join(subject_class_imgs_path, class_img_name) 
            assert osp.exists(class_img_path), f"{class_img_path = }"
            class_img = Image.open(class_img_path) 
            example["class_img"] = self.image_transforms(class_img) 
            class_prompt = f"a photo of a {random_subject}" 
            example["class_prompt"] = class_prompt 
            example["class_prompt_ids"] = self.tokenizer( 
                example["class_prompt"], 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length 
            ).input_ids 
            # print(f"{class_prompt = }") 

        assert len(example["subjects"]) == len(example["scalers"]) 
        return example 


class TextualInvDataset(Dataset): 
    def __init__(
        self, 
        args, 
        tokenizer, 
        ref_imgs_dir, 
        num_steps, 
    ): 
        self.args = args 
        self.tokenizer = tokenizer 
        self.num_steps = num_steps 
        self.ref_imgs_dir = ref_imgs_dir 

        img_transforms = [] 
        if args.resize:
            img_transforms.append(
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if args.center_crop:
            img_transforms.append(transforms.CenterCrop(args.resolution)) 
        if args.color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        # if args.h_flip:
        #     img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )


    def __len__(self): 
        return self.num_steps 


    def __getitem__(self, index): 
        pass 