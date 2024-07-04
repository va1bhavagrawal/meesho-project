import os 
import os.path as osp 
import numpy as np 
import shutil 
import math 
import cv2 
import sys 
import matplotlib.pyplot as plt 
import PIL
from PIL import Image

NUM_INSTANCES=6
DEST_PATH = "ref_imgs"
SRC_PATH = sys.argv[-1]
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

    # img = cv2.imread(img_file)

    # Open the image
    image = Image.open(img_file)

    # Create a new white background image with the same size as the original image
    background = Image.new("RGB", image.size, (0, 0, 0))

    # Paste the original image onto the white background
    background.paste(image, (0, 0), image)

    # Save the resulting image with the white background
    background.save(dest_path)

    # cv2.imwrite(dest_path, black_img)