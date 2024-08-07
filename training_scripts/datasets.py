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

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, num_samples=24, subjects=None):
        self.num_samples = num_samples 
        self.subjects = subjects 
        assert self.subjects is not None 
        # if self.subjects is None: 
        #     self.subjects = [
        #         "bnha pickup truck",
        #         "bnha motorbike",  
        #         "bnha horse", 
        #         "bnha lion", 
        #     ] 
        # self.subjects2 = [
        #     "bicycle", 
        #     "tractor", 
        #     "sports car", 
        #     "brad pitt", 
        # ]

        self.template_prompts = [
            # prompts testing if the model can follow the prompt to create an 'environment'
            "a photo of a SUBJECT on a remote country road, surrounded by rolling hills, vast open fields and tall trees", 
            "a photo of a SUBJECT on a bustling city street, surrounded by towering skyscrapers and neon lights",
            "a photo of a SUBJECT beside a field of blooming sunflowers, with snowy mountain ranges in the distance.",  
            "a SUBJECT on a tropical beach, with palm trees swaying and waves crashing on the shore", 
            "a SUBJECT in a colorful tulip field, with windmills in the background", 
        ]
        # this is just an indicator of azimuth, not the exact value 
        self.azimuths = torch.arange(num_samples)  
        self.prompt_wise_subjects, self.prompts = self.generate_prompts()
        assert len(self.prompt_wise_subjects) == len(self.prompts) 


    def generate_prompts(self):  
        prompts = []
        prompt_wise_subjects = [] 
        for subject in self.subjects: 
            for prompt in self.template_prompts:
                # if use_sks:
                #     prompt = "a sks photo of " + prompt 
                # else: 
                #     prompt = "a photo of " + prompt 
                subject_without_bnha = subject.replace("bnha", "").strip()  
                subject_without_bnha = " " + subject_without_bnha + " " 
                prompt_ = prompt.replace(f"SUBJECT", "bnha") 
                # we DO NOT want the subject to be present in the prompt text 
                assert prompt_.find(subject) == -1, f"{prompt_ = }, {prompt = }, {subject = }" 
                assert prompt_.find(subject_without_bnha) == -1, f"{prompt_ = }, {prompt = }, {subject_without_bnha = }, {subject = }" 
                prompts.append(prompt_)  
                prompt_wise_subjects.append(subject) 
        return prompt_wise_subjects, prompts  


    def __len__(self):
        return len(self.subjects) * len(self.template_prompts) * self.num_samples


    def __getitem__(self, index):
        return self.data[index] 


class DisentangleDataset(Dataset): 
    def __init__(
        self,
        args, 
        tokenizer,
    ): 
        self.args = args 
        # controlnet prompts are provided as a list, not as a filepath.
        self.tokenizer = tokenizer 

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
        if args.h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )


    def __len__(self):
        return self.args.max_train_steps  


    def __getitem__(self, index): 
        example = {} 

        # selecting the subject according to the index -- this is not randomized, every subject will get equal representation for sure  
        subject = self.args.subjects[index % len(self.args.subjects)] 
        subject_ = "_".join(subject.split()) 
        subject_ref_dir = osp.join(self.args.instance_data_dir, subject_)
        assert osp.exists(subject_ref_dir) 

        example["subject"] = subject 

        # selecting the random view for the chosen subject 
        random_ref_img = random.choice(os.listdir(subject_ref_dir))  
        angle = float(random_ref_img.split(f".jpg")[0])  
        example["scaler"] = angle 

        # choosing from the instance images, not the augmentation 
        if True:  
            example["controlnet"] = False 
            # prompt = f"a photo of a bnha {subject} in front of a dark background"  
            prompt = f"a photo of a bnha in front of a dark background"  

            example["prompt_ids"] = self.tokenizer(
                prompt, 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length, 
            ).input_ids 

            img_path = osp.join(osp.join(subject_ref_dir, random_ref_img))
            assert osp.exists(img_path) 
            img = Image.open(img_path)  

        # choosing from the controlnet augmentation 
        else: 
            example["controlnet"] = True  
            subject_angle_controlnet_dir = osp.join(self.args.controlnet_data_dir, subject_, str(angle))  
            avlble_imgs = os.listdir(subject_angle_controlnet_dir) 
            chosen_img = random.choice(avlble_imgs) 

            prompt_idx = int(chosen_img.split("___prompt")[-1].split(".jpg")[0])  
            prompt = self.args.controlnet_prompts[prompt_idx] 
            # there must be the keyword SUBJECT in the prompt, that can be replaced for the relevant subject 
            assert prompt.find("SUBJECT") != -1 
            prompt = prompt.replace("SUBJECT", f"bnha")  
            assert prompt.find("bnha") != -1 
            # assert prompt.find(subject) != -1 
            # we DO NOT want the subject to be present in the prompt text 
            assert prompt.find(f" {subject} ") == -1, f"{prompt = }, {subject = }" 
            example["prompt_ids"] = self.tokenizer(
                prompt, 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length, 
            ).input_ids 

            img_path = osp.join(subject_angle_controlnet_dir, chosen_img)
            assert osp.exists(img_path) 
            img = Image.open(img_path)   

        print(f"{prompt = }")
        print(f"{img_path = }")
        # in either case, the poseappearance embedding would be necessary 
        # in either case, the subject name in the prompt would be necessary too 
        assert prompt.find("bnha") != -1 
        # assert prompt.find(subject) != -1 
        # we DO NOT want the subject to be present in the prompt text 
        assert prompt.find(f" {subject} ") == -1, f"{prompt = }, {subject = }"  

        if not img.mode == "RGB":  
            img = img.convert("RGB") 
        example["img"] = self.image_transforms(img)  

        if self.args.with_prior_preservation: 
            subject_class_imgs_path = osp.join(self.args.class_data_dir, subject_)  
            assert len(os.listdir(subject_class_imgs_path)) == self.args.num_class_images 
            class_img_name = str(index % self.args.num_class_images).zfill(3) + ".jpg"  
            class_img_path = osp.join(subject_class_imgs_path, class_img_name) 
            assert osp.exists(class_img_path), f"{class_img_path = }"
            class_img = Image.open(class_img_path) 
            example["class_img"] = self.image_transforms(class_img) 
            class_prompt = f"a photo of a {subject}"
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt, 
                padding="do_not_pad", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length 
            ).input_ids 
            print(f"{class_prompt = }") 

        return example 