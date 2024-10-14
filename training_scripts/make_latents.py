import os 
import os.path as osp 
import torch 
import pickle 
import random 

if osp.exists("best_latents.pt"): 
    os.remove("best_latents.pt")  
seed = random.randint(0, 170904) 
with open(f"seed.pkl", "wb") as f: 
    pickle.dump(seed, f) 
# set_seed(seed) 
latents = torch.randn(1, 4, 64, 64)  
with open(f"best_latents.pt", "wb") as f: 
    torch.save(latents, f) 