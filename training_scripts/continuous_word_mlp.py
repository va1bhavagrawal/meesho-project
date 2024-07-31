"""
ADOBE CONFIDENTIAL
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe 
and its suppliers and are protected by all applicable intellectual 
property laws, including trade secret and copyright laws. 
Dissemination of this information or reproduction of this material is 
strictly forbidden unless prior written permission is obtained from Adobe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 


class continuous_word_mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(continuous_word_mlp, self).__init__()

        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x


class HotEmbedding(nn.Module):
    def __init__(self, init_embedding):
        super().__init__() 
        self.embed = torch.nn.Parameter(torch.clone(init_embedding)) 
    
    def forward(self, x): 
        return self.embed


class AppearanceEmbeddings(nn.Module):
    def __init__(self, init_embeddings: dict): 
        super().__init__() 
        self.n_embeddings = len(init_embeddings.values())  
        for key, value in init_embeddings.items(): 
            # self.embeds[key] = torch.nn.Parameter(value.detach())    
            self.register_parameter(key, torch.nn.Parameter(value.detach())) 
    
    def forward(self, name): 
        raise NotImplementedError(f"there is no forward method for this class") 
        # return getattr(self, name) 


class MergedEmbedding(nn.Module): 
    def __init__(self, pose_dim=1024, appearance_dim=1024): 
        super().__init__() 
        self.linear1 = nn.Linear(pose_dim + appearance_dim, 2048) 
        self.linear2 = nn.Linear(2048, 2048)  
        self.linear3 = nn.Linear(2048, 2048) 
        self.linear4 = nn.Linear(2048, 1024) 

        merger_state_dict = self.state_dict() 
        # for name, param in merger_state_dict.items():  
        #     merger_state_dict[name] = nn.Parameter(torch.zeros_like(param), requires_grad=True)   

        merger_state_dict["linear4.weight"] = torch.zeros_like(self.linear4.weight) 
        merger_state_dict["linear4.bias"] = torch.zeros_like(self.linear4.bias)  

        self.load_state_dict(merger_state_dict) 


    def forward(self, pose_embed, appearance_embed): 
        concat_embed = torch.cat([pose_embed, appearance_embed], dim=-1) 
        x = self.linear1(concat_embed) 
        x = F.relu(x) 
        x = self.linear2(x) 
        x = F.relu(x)  
        x = self.linear3(x) 
        x = F.relu(x) 
        x = self.linear4(x) + appearance_embed * 0.5      
        return x 