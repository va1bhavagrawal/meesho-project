from transformers import CLIPTokenizer 
import sys 

tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")  
prompt = sys.argv[-1]  
ids = tokenizer(prompt).input_ids 
print(ids) 
