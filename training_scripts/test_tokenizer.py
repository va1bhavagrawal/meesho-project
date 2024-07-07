from transformers import CLIPTokenizer 

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text = "bnha" 
tokens = tokenizer(text, return_tensors="pt") 
print(tokens) 
