import torch 
a = torch.randn((32, 4, 128, 128))  
torch.save(a, "latents.pt")  