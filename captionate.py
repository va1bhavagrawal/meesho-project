from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


import os 
import os.path as osp 
import random 

root_ref_imgs_dir = osp.join("training_data_2subjects", "ref_imgs_singlesub") 
for subject_ in os.listdir(root_ref_imgs_dir): 
    subject_dir = osp.join(root_ref_imgs_dir, subject_) 
    img_names = os.listdir(subject_dir) 
    img_names = [img_name for img_name in img_names if img_name.find("jpg") != -1] 
    img_paths = [osp.join(subject_dir, img_name) for img_name in img_names] 
    chosen_img_paths = random.sample(img_paths, 2) 
    
    preds = predict_step([chosen_img_paths]) # ['a woman in a hospital bed with a woman in a hospital bed']
    print(f"{preds}")
