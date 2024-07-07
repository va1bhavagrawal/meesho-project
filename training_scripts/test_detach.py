import torch

# Create a tensor with gradients
x = torch.tensor([1, 2, 3]).float()
x.requires_grad_(True)

# Detach the tensor
y = x.detach()

print(f"before value change:")
print(x)
print(y)

# Modify the detached tensor
y[0] = 10

# The original tensor is not affected
print(x)  
print(y)  