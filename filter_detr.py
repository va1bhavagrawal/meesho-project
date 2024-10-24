import torch
import os 
import os.path as osp 
import shutil 

import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.ops import box_iou

# Load a pre-trained object detection model (e.g., Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transform to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to load image and run object detection
def detect_objects(image_path, model, threshold=0.5):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Transform the image
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Detect objects
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Filter out predictions below the threshold
    scores = predictions[0]['scores'].numpy()
    boxes = predictions[0]['boxes'].numpy()
    filtered_boxes = boxes[scores > threshold]
    
    return filtered_boxes

# Function to calculate best matching IoU between two sets of boxes
def match_boxes_and_calculate_iou(ref_boxes, gen_boxes, iou_threshold=0.5):
    if len(ref_boxes) == 0 or len(gen_boxes) == 0:
        return 0, 0  # No objects detected in one of the images
    
    # Compute IoU between all pairs of boxes
    iou_matrix = box_iou(torch.tensor(ref_boxes), torch.tensor(gen_boxes)).numpy()
    
    # Match boxes greedily based on highest IoU
    matches = []
    matched_ref_idx = set()
    matched_gen_idx = set()
    
    # Find the best matches
    for i in range(len(ref_boxes)):
        if len(np.where(iou_matrix[i] >= iou_threshold)[0]) > 0:
            best_match_idx = np.argmax(iou_matrix[i])
            if best_match_idx not in matched_gen_idx:
                matches.append(iou_matrix[i][best_match_idx])
                matched_ref_idx.add(i)
                matched_gen_idx.add(best_match_idx)
    
    avg_iou = np.mean(matches) if matches else 0
    return avg_iou, len(matches)

# Function to evaluate the quality of the generated images
def evaluate_images(generated_folder, model, threshold=0.5, iou_threshold=0.8):
    from os import listdir
    from os.path import join

    # reference_images = sorted([join(reference_folder, img) for img in listdir(reference_folder)])
    generated_images = sorted([join(generated_folder, img) for img in listdir(generated_folder)])
    
    good_generations = []
    
    for gen_img_path in generated_images:
        # Detect objects in both reference and generated images
        subjects_string, _ = osp.split(gen_img_path)  
        _, subjects_string = osp.split(subjects_string) 
        gen_img_name = osp.basename(gen_img_path) 
        ref_img_name = "__".join(gen_img_name.split("__")[:-2] + gen_img_name.split("__")[-1:])  
        ref_img_path = osp.join(f"training_data_2010", f"ref_imgs_2subjects", subjects_string, ref_img_name)  
        assert osp.exists(ref_img_path), f"{ref_img_path = }" 
        # continue 
        print(f"doing {gen_img_path}")
        ref_boxes = detect_objects(ref_img_path, model, threshold)
        gen_boxes = detect_objects(gen_img_path, model, threshold)


        if len(ref_boxes) != len(gen_boxes): 
            continue 
        
        # Match boxes and calculate the IoU
        avg_iou, num_matches = match_boxes_and_calculate_iou(ref_boxes, gen_boxes, iou_threshold)
        
        # If the average IoU of matched boxes is greater than the threshold, consider it a good generation
        if avg_iou >= iou_threshold and num_matches == len(ref_boxes):  # All objects should be matched
            good_generations.append(gen_img_path)
    
    return good_generations

# Define your folders here
reference_folder = '2010_2obj_canny_0.5_20/sedan__teddy'

# Evaluate the generated images
good_generations = evaluate_images(reference_folder, model)

# Output the good generations
print("Good generations based on IoU threshold:")
filtered_dir = f"faster_rcnn_filtered"
os.makedirs(filtered_dir)  
for gen in good_generations:
    assert osp.exists(gen) 
    new_path = osp.join()
    shutil.copy(gen, osp.join(filtered_dir, osp.basename(gen)))  