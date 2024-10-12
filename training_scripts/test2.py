import torch 
import torch.nn.functional as F 
import numpy as np 
import time 
import matplotlib.pyplot as plt 

h = 32  
w = 32  

def make_voronoi_attention_mask(centroids, attention_map_size, temperature, infinity): 
    i_grid, j_grid = torch.meshgrid(torch.linspace(0, 1, attention_map_size[0]), torch.linspace(0, 1, attention_map_size[1]), indexing="ij") 
    ij_grid = torch.stack((i_grid, j_grid), dim=-1).to(centroids.device)  
    dists = torch.sum((ij_grid.unsqueeze(0) - centroids.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)  
    spatial_dim = dists.shape[-1] 
    if len(centroids) == 2:  
        masks = torch.stack([dists[0] - dists[1], dists[1] - dists[0]], dim=0) 
    elif len(centroids) == 1: 
        masks = dists[0].unsqueeze(0)  
    else: 
        raise NotImplementedError() 
    masks = F.sigmoid(temperature * masks.flatten(1, ))  
    minval = torch.min(masks, dim=-1).values.unsqueeze(-1)  
    maxval = torch.max(masks, dim=-1).values.unsqueeze(-1)  
    masks = (masks - minval) / (maxval - minval)  
    masks = masks.reshape(masks.shape[0], spatial_dim, spatial_dim) 
    for mask in masks: 
        assert torch.max(mask) == 1 and torch.min(mask) == 0 
    masks = (1 - masks) * (-infinity) 
    return masks 

while True: 
    point1 = torch.randint(0, h, (2, )) / h 
    point2 = torch.randint(0, h, (2, )) / h  
    centroids = torch.stack([point1, point2], dim=0) 
    masks = make_voronoi_attention_mask(centroids, (h, w), 100.0, 15)   
    fig, axs = plt.subplots(1, 2, figsize=(20, 5)) 
    i1, j1 = centroids[0] * h  
    i2, j2 = centroids[1] * h  
    axs[0].imshow(masks[0].numpy()) 
    axs[0].scatter([j1], [i1]) 
    axs[0].scatter([j2], [i2]) 
    axs[1].imshow(masks[1].numpy()) 
    axs[1].scatter([j1], [i1]) 
    axs[1].scatter([j2], [i2]) 
    plt.savefig(f"mask.jpg") 
    plt.close() 
    time.sleep(1) 

