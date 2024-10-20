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
import sys 

import os 
import os.path as osp 

from infer_online import UNIQUE_TOKENS 

import pickle 

MAX_SUBJECTS_PER_EXAMPLE = 2  
RAW_IMG_SIZE = 1024 


class PosedSubjectsDataset(Dataset): 
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

        unique_strings = []  
        for asset_idx in range(2):  
            unique_string_subject = "" 
            for token_idx in range(self.args.merged_emb_dim // 1024): 
                unique_string_subject = unique_string_subject + f"{UNIQUE_TOKENS[f'{asset_idx}_{token_idx}']}"  
            unique_string_subject = unique_string_subject.strip() 
            unique_strings.append(unique_string_subject) 

        self.examples = {}  

        if self.args.use_ref_images: 
            self.subjects_combs_ref = {}  
            for ref_imgs_dir in ref_imgs_dirs: 
                print(f"preparing images in {ref_imgs_dir}") 
                self.examples[ref_imgs_dir] = [] 
                if not osp.exists(ref_imgs_dir): 
                    continue 
                self.subjects_combs_ref[ref_imgs_dir] = [] 
                subjects_combs_ = os.listdir(ref_imgs_dir) 
                for subjects_comb_ in subjects_combs_: 
                    subjects_path = osp.join(ref_imgs_dir, subjects_comb_) 
                    self.subjects_combs_ref[ref_imgs_dir].append(subjects_comb_)  

                    # put images from this directory into the self.examples 
                    filenames = os.listdir(subjects_path) 
                    img_names = [filename for filename in filenames if filename.find("jpg") != -1] 
                    pkl_names = [img_name.replace("jpg", "pkl") for img_name in img_names]
                    img_paths = [osp.join(subjects_path, img_name) for img_name in img_names] 
                    pkl_paths = [osp.join(subjects_path, pkl_name) for pkl_name in pkl_names] 
                    for img_path in img_paths: 
                        assert osp.exists(img_path) 
                    for pkl_path in pkl_paths: 
                        assert osp.exists(pkl_path) 
                    for img_path, pkl_path in zip(img_paths, pkl_paths): 
                        img_name = osp.basename(img_path) 
                        example = {"controlnet": False}  
                        example["subjects"] = subjects_comb_.split("__") 
                        example["img_path"] = img_path 
                        with open(pkl_path, "rb") as f: 
                            pkl_data = pickle.load(f) 

                        all_2d_x = [] 
                        all_2d_y = [] 
                        all_bboxes = [] 
                        # print(f"{pkl_data = }")
                        # sys.exit(0) 
                        for asset_idx in range(len(pkl_data.keys())): 
                            if len(pkl_data[f"obj{asset_idx+1}"]) == 0: 
                                continue 
                            bbox = pkl_data[f"obj{asset_idx+1}"]["bbox"] 
                            all_2d_x.append((bbox[0] + bbox[2]) / (2 * RAW_IMG_SIZE))  
                            all_2d_y.append((bbox[1] + bbox[3]) / (2 * RAW_IMG_SIZE))  
                            all_bboxes.append(torch.tensor(bbox) / RAW_IMG_SIZE)  
                        example["bboxes"] = all_bboxes 
                        example["2d_xs"] = all_2d_x 
                        example["2d_ys"] = all_2d_y  

                        subjects_data = img_name.split("__") 
                        subjects_data = subjects_data[:-1] 
                        all_x = []  
                        all_y = [] 
                        all_z = [] 
                        all_a = [] 
                        assert len(subjects_data) == len(example["subjects"]), f"{subjects_data = }, {example['subjects'] = }" 
                        assert len(subjects_data) <= MAX_SUBJECTS_PER_EXAMPLE 
                        for asset_idx in range(len(example["subjects"])): 
                            one_subject_data = subjects_data[asset_idx] 
                            x, y, z, a = one_subject_data.split("_") 
                            all_x.append(float(x)) 
                            all_y.append(float(y)) 
                            all_z.append(float(z)) 
                            all_a.append(float(a)) 
                        example["scalers"] = all_a 


                        template_prompt = "a photo of PLACEHOLDER in a dark studio with white lights" 
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
                            for asset_idx in range(len(example["subjects"])): 
                                assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                                template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {example['subjects'][asset_idx]}") 

                        example["prompt"] = template_prompt   
                        example["prompt_ids"] = self.tokenizer(
                            example["prompt"], 
                            padding="do_not_pad", 
                            truncation=True, 
                            max_length=self.tokenizer.model_max_length, 
                        ).input_ids 

                        self.examples[ref_imgs_dir].append(example) 


        if self.args.use_controlnet_images: 
            self.subjects_combs_controlnet = {}  
            for controlnet_imgs_dir in controlnet_imgs_dirs: 
                print(f"preparing images in {controlnet_imgs_dir}") 
                self.examples[controlnet_imgs_dir] = [] 
                if not osp.exists(controlnet_imgs_dir): 
                    continue 
                self.subjects_combs_controlnet[controlnet_imgs_dir] = [] 
                subjects_combs_ = os.listdir(controlnet_imgs_dir) 
                for subjects_comb_ in subjects_combs_: 
                    subjects_path = osp.join(controlnet_imgs_dir, subjects_comb_) 
                    self.subjects_combs_controlnet[controlnet_imgs_dir].append(subjects_comb_)  
                    img_names = os.listdir(subjects_path) 
                    for img_name in img_names: 
                        assert img_name.find("jpg") != -1 
                    img_paths = [osp.join(subjects_path, img_name) for img_name in img_names] 
                    for img_path in img_paths: 
                        img_name = osp.basename(img_path) 
                        assert osp.exists(img_path) 
                        example = {"controlnet": True}  
                        example["subjects"] = subjects_comb_.split("_") 
                        pkl_path = None 
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
                        example["2d_xs"] = all_2d_x 
                        example["2d_ys"] = all_2d_y 
                        example["bboxes"] = all_bboxes 

                        subjects_data = img_name.split("__") 
                        subjects_data = subjects_data[:-1] 
                        whichprompt = subjects_data[-1] 
                        prompt_idx = int(whichprompt.replace("prompt", "").strip()) 
                        subjects_data = subjects_data[:-1] 
                        all_x = [] 
                        all_y = [] 
                        all_z = [] 
                        all_a = [] 
                        for asset_idx in range(len(example["subjects"])): 
                            one_subject_data = subjects_data[asset_idx] 
                            x, y, z, a = one_subject_data.split("_") 
                            all_x.append(float(x))
                            all_y.append(float(y)) 
                            all_z.append(float(z)) 
                            all_a.append(float(a)) 
                        example["scalers"] = all_a 


                        template_prompt = self.args.controlnet_prompts[prompt_idx] 
                        placeholder_text = "a SUBJECT0 " 
                        for asset_idx in range(1, len(example["subjects"])):  
                            placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} "  
                        placeholder_text = placeholder_text.strip() 
                        template_prompt = template_prompt.replace("PLACEHOLDER", placeholder_text) 
                        if not self.args.include_class_in_prompt: 
                            for asset_idx in range(len(example["subjects"])): 
                                assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                                template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{example['subjects'][asset_idx]}")  
                        else: 
                            for asset_idx in range(len(example["subjects"])): 
                                assert template_prompt.find(f"SUBJECT{asset_idx}") != -1 
                                template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {example['subjects'][asset_idx]}") 

                        example["prompt"] = template_prompt 
                        example["prompt_ids"] = self.tokenizer( 
                            example["prompt"], 
                            padding="do_not_pad", 
                            truncation=True, 
                            max_length=self.tokenizer.model_max_length, 
                        ).input_ids 

                        self.examples[controlnet_imgs_dir].append(example) 
        
        # self.num_distinct_examples = 0  
        # if self.args.use_ref_imgs: 
        #     for ref_imgs_dir in ref_imgs_dirs: 
        #         self.num_distinct_examples += len(self.examples[ref_imgs_dir]) 
        # if self.args.use_controlnet_imgs: 
        #     for controlnet_imgs_dir in controlnet_imgs_dirs: 
        #         self.num_distinct_examples += len(self.examples[controlnet_imgs_dir]) 

        self.examples_list = [] 
        if self.args.use_ref_images: 
            for ref_imgs_dir in ref_imgs_dirs: 
                self.examples_list += self.examples[ref_imgs_dir]  
        if self.args.use_controlnet_images:  
            for controlnet_imgs_dir in controlnet_imgs_dirs: 
                self.examples_list += self.examples[controlnet_imgs_dir]  


    def __len__(self):
        # return self.args.max_train_steps  
        # a window of size 16 is there to prevent border cases 
        return self.num_steps + 16   


    def __getitem__(self, index): 
        index = index % len(self.examples_list) 
        example = self.examples_list[index] 
        img_path = example["img_path"] 
        img = Image.open(img_path) 
        img = img.convert("RGB") 
        example["img"] = self.image_transforms(img) 
        return example 



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