import torch
from PIL import Image
import clip

# Load the pre-trained CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the input image
image = Image.open("input_image.jpg")

# Preprocess the image
image = preprocess(image).unsqueeze(0).to(device)

# Encode the text prompt
text = clip.tokenize(["a photo of a dog"]).to(device)

# Encode the image
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Calculate the similarity
similarity = torch.cosine_similarity(image_features, text_features, dim=-1).item()
print(f"CLIP Similarity: {similarity:.2f}")