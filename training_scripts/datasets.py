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


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, num_samples=24, use_sks=True):
        self.num_samples = num_samples 
        self.subjects = [
            "bnha pickup truck",
            "motorbike", 
        ]

        self.template_prompts = [
            # prompts testing if the model can follow the prompt to create an 'environment'
            "a SUBJECT parked on a remote country road, surrounded by rolling hills, vast open fields and tall trees", 
            "a SUBJECT parked on a bustling city street, surrounded by towering skyscrapers and neon lights",
            "a SUBJECT beside a field of blooming sunflowers." 
        ]
        # this is just an indicator of azimuth, not the exact value 
        self.azimuths = torch.arange(num_samples)  
        self.prompts = self.generate_prompts(use_sks)


    def generate_prompts(self, use_sks=True):
        prompts = []
        for subject in self.subjects: 
            for prompt in self.template_prompts:
                if use_sks:
                    prompt = "a sks photo of " + prompt 
                else: 
                    prompt = "a photo of " + prompt 
                prompt_ = prompt.replace(f"SUBJECT", subject)
                prompts.append(prompt_)  
        return prompts  


    def __len__(self):
        return len(self.subjects) * len(self.template_prompts) * self.num_samples


    def __getitem__(self, index):
        return self.data[index] 