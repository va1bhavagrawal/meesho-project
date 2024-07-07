import os 
import os.path as osp 
import numpy as np 
import shutil 
import math 
import cv2 
import sys 
import matplotlib.pyplot as plt 
from PIL import Image

NUM_INSTANCES=36
SRC_PATH = sys.argv[-1]
DEST_PATH = f"ref_imgs_{SRC_PATH}"
if osp.exists(DEST_PATH): 
    shutil.rmtree(DEST_PATH) 
os.makedirs(DEST_PATH) 


for idx in range(NUM_INSTANCES):  
    img_file = osp.join(SRC_PATH, str(idx).zfill(3) + ".png") 
    np_file = osp.join(SRC_PATH, "xyz_" + str(idx).zfill(3) + ".npy") 
    np_data = np.load(np_file)
    x, y, _ = np_data
    angle = math.atan2(y, x)
    if angle < 0:
        angle = angle + 2 * math.pi
    dest_path = osp.join(DEST_PATH, f"{angle}_.jpg")
    print(f"{dest_path = }")

    # Open the image
    image = Image.open(img_file)

    # Crop the image to the minimum bounding box
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)

    # Add 10% padding to the image
    width, height = cropped_image.size
    padding = int(0.1 * min(width, height))
    padded_image = Image.new("RGB", (width + 2*padding, height + 2*padding), (0, 0, 0))
    padded_image.paste(cropped_image, (padding, padding))

    # Resize the largest dimension to 512
    max_dimension = max(padded_image.size)
    if max_dimension > 512:
        resize_factor = 512 / max_dimension
        new_size = (int(padded_image.size[0] * resize_factor), int(padded_image.size[1] * resize_factor))
        resized_image = padded_image.resize(new_size, resample=Image.BICUBIC)
    else:
        resized_image = padded_image

    # Add padding to make the final image size 512x512
    final_image = Image.new("RGB", (512, 512), (0, 0, 0))
    left = (512 - resized_image.size[0]) // 2
    top = 3 * (512 - resized_image.size[1]) // 4 
    final_image.paste(resized_image, (left, top))

    # Save the resulting image
    final_image.save(dest_path)
