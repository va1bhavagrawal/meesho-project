import transformers 
from transformers import CLIPTextModel, CLIPTokenizer  

MODEL_NAME = "stabilityai/stable-diffusion-2-1"

text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")  
inp_embeds = text_encoder.get_input_embeddings().weight 
print(f"{inp_embeds.shape = }")
prompt = "a photo of a man" 
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")  
tokens = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()  
print(f"{tokens.shape = }")
embeddings = text_encoder(tokens)[0]  
print(f"{embeddings.shape = }")