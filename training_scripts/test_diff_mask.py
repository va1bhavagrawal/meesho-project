import torch 
import numpy as np  
import matplotlib.pyplot as plt 

INFINITY = 1e4 

def generate_attention_scores_mask(bbox, attention_map_size, temperature=1.0):
    """
    Generates a stable attention mask based on predicted bounding box.
    
    bbox: Tensor of shape [batch_size, 4] (x_min, y_min, x_max, y_max)
    attention_map_size: (H, W) of the attention map to be masked
    temperature: controls the softness of the mask (lower = sharper)
    large_neg: large negative value to penalize areas outside the bounding box

    Returns:
    - A mask of shape [batch_size, H, W], with values between large_neg and 0.
    """
    batch_size = bbox.shape[0]
    H, W = attention_map_size

    # Create a grid of coordinates
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
    y_grid, x_grid = y_grid.to(bbox.device), x_grid.to(bbox.device)

    # Reshape the grid to [1, H, W] so it can broadcast with bbox
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)

    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3] 

    # Create soft masks for x and y dimensions using sigmoid for differentiability
    x_mask = torch.sigmoid(temperature * (x_grid - x_min.unsqueeze(1).unsqueeze(2))) * \
             torch.sigmoid(temperature * (x_max.unsqueeze(1).unsqueeze(2) - x_grid)) 

    y_mask = torch.sigmoid(temperature * (y_grid - y_min.unsqueeze(1).unsqueeze(2))) * \
             torch.sigmoid(temperature * (y_max.unsqueeze(1).unsqueeze(2) - y_grid)) 

    x_mask = (x_mask - torch.min(x_mask)) / (torch.max(x_mask) - torch.min(x_mask)) 
    y_mask = (y_mask - torch.min(y_mask)) / (torch.max(y_mask) - torch.min(y_mask)) 
    x_mask = torch.max(x_mask, torch.tensor(1e-6)) 
    y_mask = torch.max(y_mask, torch.tensor(1e-6))  

    plt.imshow(x_mask.squeeze().detach().cpu().numpy()) 
    plt.colorbar() 
    plt.savefig(f"x_mask.jpg") 
    plt.close() 

    plt.imshow(y_mask.squeeze().detach().cpu().numpy()) 
    plt.colorbar() 
    plt.savefig(f"y_mask.jpg") 
    plt.close() 

    # Combine x and y masks to create a final soft mask
    soft_mask = x_mask * y_mask

    # Rescale the mask to be in [large_neg, 0]
    attention_mask = (1 - soft_mask) * (-INFINITY) 


    plt.imshow(attention_mask.detach().squeeze().cpu().numpy()) 
    plt.colorbar() 
    plt.savefig(f"attention_mask.jpg") 
    plt.close() 

    return attention_mask  


temperature = 100.0  
bbox = torch.tensor([0.5, 0.5, 0.75, 1.00]).unsqueeze(0)  
attention_map_size = (32, 32)  
mask = generate_attention_scores_mask(bbox, attention_map_size, temperature=temperature).squeeze()  
mask = mask.cpu().numpy() 
print(f"{np.min(mask) = }")
print(f"{np.max(mask) = }")
# mask = np.log(mask) 
plt.imshow(mask) 
plt.colorbar() 
plt.title(f"{temperature= }")
plt.savefig(f"stable_mask_{temperature}.jpg") 