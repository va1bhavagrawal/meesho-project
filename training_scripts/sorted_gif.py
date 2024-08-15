import os
from PIL import Image, ImageDraw, ImageFont
import imageio
import cv2 
import numpy as np 
from utils import create_gif 


def make_single_folder_gif(subject_): 
    # Gather images
    folder_path = os.path.join("sorted_experiment", subject_) 
    # gathered_images = sorted([f for f in sorted(os.listdir(folder_path)) if f.endswith('.jpg')])
    img_names = os.listdir(folder_path) 
    img_names = [img_name for img_name in img_names if img_name.find("jpg") != -1] 
    n_imgs = len(img_names) 
    img_ids = np.arange(0, n_imgs, 2)  
    gathered_images = [str(img_idx).zfill(4) + ".jpg" for img_idx in img_ids] 

    # Create a list to hold the images for the GIF
    gif_images = []

    caption_height = 50 

    # Load each image, add a caption, and append to the gif_images list
    for img_name in gathered_images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        # Create a draw object
        
        # Add caption (the number of the image)
        caption = f"{subject_} :: {img_name.split('.')[0]}"   # Get the base name without extension
        # # text_width, text_height = draw.textsize(caption, font=font)
        # bbox = draw.textbbox((0, 0), caption, font=font)  # For Pillow 9.0.0 and later
        # text_width = bbox[2] - bbox[0]
        # text_height = bbox[3] - bbox[1]
        
        # # Position the text at the bottom center
        # text_position = ((img.width - text_width) // 2, img.height - text_height - 10)
        
        # # Draw the text on the image
        # draw.text(text_position, caption, fill="white", font=font)

        caption_img = np.ones((caption_height, np.array(img).shape[1], 3), dtype=np.uint8) * 255
        # Put the caption text on the blank image
        cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.vconcat([np.array(img).astype(np.uint8), caption_img]) 
        
        # Append the modified image to the list
        gif_images.append(img)

    # Save the images as a GIF
    gif_path = f"{subject_}.gif" 
    # print(f"{gif_path = }") 
    # imageio.mimsave(gif_path, gif_images, duration=0.5)  # Adjust duration for speed
    gif_images = [Image.fromarray(img) for img in gif_images] 
    create_gif(gif_images, gif_path, 1)  


for subject_ in os.listdir("sorted_experiment"): 
    print(f"doing {subject_}") 
    make_single_folder_gif(subject_)  
