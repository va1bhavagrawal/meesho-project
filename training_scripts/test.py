from transformers import CLIPTextModel, CLIPTokenizer 
import torch 

model_name = "stabilityai/stable-diffusion-2-1"

tokenizer = CLIPTokenizer.from_pretrained(
    model_name, 
    subfolder="tokenizer", 
)

text_encoder = CLIPTextModel.from_pretrained(
    model_name,
    subfolder="text_encoder", 
)

learned_embed = torch.randn(1024, requires_grad=True)  
print(f"{learned_embed.shape = }")

prompt = "a bnha photo of a truck" 
tokens = tokenizer(
    prompt, 
    padding="max_length", 
).input_ids  

inp_embeds = torch.clone(text_encoder.get_input_embeddings().weight).detach()  
text_encoder.get_input_embeddings().weight = torch.nn.Parameter(inp_embeds, requires_grad=False)  
print(f"before replacement, {text_encoder.get_input_embeddings().weight.requires_grad = }")
print(f"{text_encoder.get_input_embeddings().weight.shape = }")
text_encoder.get_input_embeddings().weight[49336] = learned_embed 
print(f"after replacement, {text_encoder.get_input_embeddings().weight.requires_grad = }")
# print(f"{tokens = }")
# print(f"{type(tokens) = }")
tokens = torch.tensor(tokens).unsqueeze(0)  
encoder_hidden_states = text_encoder(tokens)[0]  
print(f"{encoder_hidden_states.shape = }")
mean = torch.mean(encoder_hidden_states) 
mean.backward() 

assert learned_embed.grad is not None 
print(f"{learned_embed.grad = }")
assert torch.sum(learned_embed.grad) 