import torch 
a = torch.load("best_latents1.pt") 
b = torch.load("best_latents2.pt") 
c = torch.load("best_latents3.pt") 
assert not torch.allclose(a, b) 
assert not torch.allclose(b, c)  
assert not torch.allclose(c, a)   
