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

from diffusers.models.embeddings import GaussianFourierProjection 


# class GaussianFourierProjection(nn.Module):
#     """Gaussian Fourier embeddings for noise levels."""

#     def __init__(
#         self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
#         self.log = log
#         self.flip_sin_to_cos = flip_sin_to_cos

#         if set_W_to_weight:
#             # to delete later
#             self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

#             self.weight = self.W

#     def forward(self, x):
#         if self.log:
#             x = torch.log(x)

#         x_proj = x[:, None] * self.weight[None, :] * 2 * torch.pi 
#         assert not torch.any(torch.isnan(x_proj)) 

#         if self.flip_sin_to_cos:
#             out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
#         else:
#             out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
#         return out


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
        # raise NotImplementedError(f"there is no forward method for this class") 
        return getattr(self, name) 


class MergedEmbedding(nn.Module): 
    def __init__(self, skip_conn, pose_dim, appearance_dim, output_dim): 
        super().__init__() 
        self.linear1 = nn.Linear(pose_dim + appearance_dim, 2048) 
        self.linear2 = nn.Linear(2048, 2048)  
        self.linear3 = nn.Linear(2048, output_dim)  
        self.skip_conn = skip_conn 
        self.output_dim = output_dim 

        merger_state_dict = self.state_dict() 
        # for name, param in merger_state_dict.items():  
        #     merger_state_dict[name] = nn.Parameter(torch.zeros_like(param), requires_grad=True)   

        merger_state_dict["linear3.weight"] = torch.zeros_like(self.linear3.weight) 
        merger_state_dict["linear3.bias"] = torch.zeros_like(self.linear3.bias)  

        self.load_state_dict(merger_state_dict)

    def forward(self, pose_embed, appearance_embed): 
        concat_embed = torch.cat([pose_embed, appearance_embed], dim=-1) 
        x = self.linear1(concat_embed) 
        x = F.relu(x) 
        x = self.linear2(x) 
        x = F.relu(x) 
        if self.skip_conn: 
            x = self.linear3(x) + appearance_embed.repeat(1, 1, self.output_dim // 1024)  
        else: 
            x = self.linear3(x) 

        return x 


class PoseEmbedding(nn.Module): 
    def __init__(self, output_dim): 
        super().__init__() 
        self.input_dim = 1  
        self.output_dim = output_dim 
        self.linear1 = nn.Linear(output_dim, output_dim)  
        self.linear2 = nn.Linear(output_dim, output_dim)  
        self.linear3 = nn.Linear(output_dim, output_dim) 
        self.gaussian_fourier_embedding = GaussianFourierProjection(output_dim // 2, log=False)  


    def forward(self, x):  
        x = self.gaussian_fourier_embedding(x) 
        # print(f"{torch.min(x) = }, {torch.max(x) = }")
        # assert not torch.any(torch.isinf(x)) 
        # assert not torch.any(torch.isnan(x)) 
        # the output of gaussian fourier projection is of shape (B, output_dim) 
        x = self.linear1(x) 
        x = F.relu(x) 
        x = self.linear2(x) 
        x = F.relu(x) 
        x = self.linear3(x) 
        return x 


class PoseLocationEmbedding(nn.Module): 
    def __init__(self, fourier_embedding_dim, output_dim): 
        super().__init__() 
        self.input_dim = 1  
        self.fourier_embedding_dim = fourier_embedding_dim 
        self.output_dim = output_dim 
        self.linear1 = nn.Linear(3 * fourier_embedding_dim, output_dim)  
        self.linear2 = nn.Linear(output_dim, output_dim)  
        self.linear3 = nn.Linear(output_dim, output_dim) 
        self.gaussian_fourier_embedding = GaussianFourierProjection(fourier_embedding_dim // 2, log=False)  
        self.gaussian_fourier_embedding_x = GaussianFourierProjection(fourier_embedding_dim // 2, log=False)   
        self.gaussian_fourier_embedding_y = GaussianFourierProjection(fourier_embedding_dim // 2, log=False)   


    def forward(self, a, x, y):   
        azimuth_embedding = self.gaussian_fourier_embedding(a) 
        x_embedding = self.gaussian_fourier_embedding(x) 
        y_embedding = self.gaussian_fourier_embedding(y) 
        merged_embedding = torch.cat([azimuth_embedding, x_embedding, y_embedding], -1)  
        # print(f"{torch.min(x) = }, {torch.max(x) = }")
        # assert not torch.any(torch.isinf(x)) 
        # assert not torch.any(torch.isnan(x)) 
        # the output of gaussian fourier projection is of shape (B, output_dim) 
        x = self.linear1(merged_embedding) 
        x = F.relu(x) 
        x = self.linear2(x) 
        x = F.relu(x) 
        x = self.linear3(x) 
        return x 