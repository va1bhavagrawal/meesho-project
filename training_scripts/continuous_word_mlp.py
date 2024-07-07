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


class MergerMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(2 * input_size, 3 * input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.fc3 = nn.Linear(output_size, output_size)

    def forward(self, pose_emb, appearance_emb): 
        combined_emb = torch.cat([pose_emb, appearance_emb], dim=-1)
        x = F.relu(self.fc1(combined_emb)) 
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x)) 
        return x
