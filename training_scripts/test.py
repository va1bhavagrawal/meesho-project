import torch 
import numpy as np 
import time 

h = 512 
w = 512 

i_grid, j_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij") 
print(f"{i_grid = }") 
print(f"{j_grid = }")

temp = 0.00001   

while True: 

    point1 = torch.randint(0, 512, (2, ))
    point2 = torch.randint(0, 512, (2, )) 
    centroids = torch.tensor(torch.stack([point1, point2], dim=0)).to(torch.long)  
    ij_grid = torch.stack((i_grid, j_grid), dim=-1) 
    print(f"{ij_grid.shape = }") 
    print(f"{centroids.shape = }")
    dists = torch.sum((ij_grid.unsqueeze(0) - centroids.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)  
    print(f"{dists.shape = }") 
    spatial_dim = dists.shape[-1] 
    mask = torch.argmin(dists.flatten(1,), dim=0).reshape(spatial_dim, spatial_dim) 
    print(f"{mask = }") 

    import matplotlib.pyplot as plt 
    plt.imshow(mask) 
    # viridis = plt.get_cmap('viridis') 
    i1, j1 = point1.numpy() 
    i2, j2 = point2.numpy() 
    plt.scatter([j1], [i1])  
    plt.scatter([j2], [i2])  
    time.sleep(1) 
    plt.savefig(f"mask.jpg") 
    plt.close() 