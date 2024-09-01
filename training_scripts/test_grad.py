import torch 

a = torch.tensor([1.9, 1.9], requires_grad=True)  
# with torch.no_grad(): 
#     a[0] = 3 * a[0]  
#     a[1] = 3 * a[0] 
# a[0] = 3 * a[0] 
b = 3 * a 

loss = (torch.tensor(8.0) - torch.sum(b)) 
loss.backward() 

# print(f"{b.grad = }")
print(f"{a.grad = }")