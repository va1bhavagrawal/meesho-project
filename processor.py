from PIL import Image

def add_black_background_with_padding(input_image_path, output_image_path):
    # Open the input image
    img = Image.open(input_image_path).convert("RGBA")
    
    # Get the size of the image
    width, height = img.size
    
    # Create a new image with a black background
    # Calculate the padding
    padding = int(0.5 * min(width, height))
    
    # Calculate new size with padding
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    
    # Create a new black image with the new size
    new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 255))
    
    # Calculate the y-coordinate for placing the object at 3/4 height
    y_offset = int(0.75 * new_height - height / 2)
    
    # Paste the original image onto the black background
    new_img.paste(img, (padding, y_offset), img)
    
    # Convert to RGB before saving as JPEG
    new_img = new_img.convert("RGB")
    
    # Save the output image
    new_img.save(output_image_path, "JPEG")

if __name__ == "__main__":
    # input_image_path = "input_image.png"  # Change this to your input image path
    # output_image_path = "output_image.jpg"  # Change this to your desired output path
    # add_black_background_with_padding(input_image_path, output_image_path)
    # print(f"Processed image saved as {output_image_path}")
    import os 
    import math 
    import os.path as osp 
    import numpy as np  

    MULTIOBJ_DIR = "./multiobject_dataset/" 
    OUTPUT_DIR = "./ref_multiobject_imgs/"
    for subject_ in os.listdir(MULTIOBJ_DIR):  
        subject = " ".join(subject_.split("_")) 
        subject_path = osp.join(MULTIOBJ_DIR, subject_)  
        img_names = os.listdir(subject_path) 
        img_names = sorted([img_name for img_name in img_names if img_name.find(f"png") != -1])  
        angles = np.arange(len(img_names)) / len(img_names) * 2 * math.pi  
        img_paths = [osp.join(subject_path, img_name) for img_name in img_names] 
        os.makedirs(osp.join(OUTPUT_DIR, subject_), exist_ok=True) 
        output_img_paths = [osp.join(OUTPUT_DIR, subject_, str(angle) + ".jpg") for angle in angles]  
        for img_path, output_path in zip(img_paths, output_img_paths): 
            add_black_background_with_padding(img_path, output_path) 