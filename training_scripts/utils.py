import cv2 
import numpy as np 
from PIL import Image 
import datetime

# def create_image_with_captions(images, captions):
#     # Load images
#     # images = [cv2.imread(path) for path in image_paths]
#     images = [np.array(image).astype(np.uint8) for image in images] 

#     # Check if all images are loaded
#     if any(img is None for img in images):
#         raise ValueError("One or more images could not be loaded. Check the paths.")

#     # Resize images to the same height for proper concatenation
#     height = min(img.shape[0] for img in images)
#     images_resized = [cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height)) for img in images]

#     # Create a blank image for captions
#     caption_height = 50  # Height for the caption area
#     caption_images = []

#     for caption in captions:
#         # Create a blank image for the caption
#         caption_img = np.ones((caption_height, images_resized[0].shape[1], 3), dtype=np.uint8) * 255
#         # Put the caption text on the blank image
#         cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
#         caption_images.append(caption_img)

#     # Concatenate images and captions
#     final_images = [cv2.vconcat([img, caption]) for img, caption in zip(images_resized, caption_images)]
#     final_image = cv2.hconcat(final_images)

#     return Image.fromarray(final_image) 

import cv2
import numpy as np
from PIL import Image

def create_image_with_captions(image_rows, captions):
    # Check if the number of rows matches the number of caption rows
    if len(image_rows) != len(captions):
        raise ValueError("The number of caption rows must match the number of image rows.")

    # Load images and check if they are valid
    images_resized_rows = []
    for row in image_rows:
        images = [np.array(image).astype(np.uint8) for image in row]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded. Check the paths.")
        
        # Resize images to the same height for proper concatenation
        height = min(img.shape[0] for img in images)
        images_resized = [cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height)) for img in images]
        images_resized_rows.append(images_resized)

    # Create a blank image for captions
    caption_height = 50  # Height for the caption area
    caption_images = []

    for caption_row in captions:
        caption_images_row = []
        for caption in caption_row:
            # Create a blank image for the caption
            caption_img = np.ones((caption_height, images_resized_rows[0][0].shape[1], 3), dtype=np.uint8) * 255
            # Put the caption text on the blank image
            cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            caption_images_row.append(caption_img)
        caption_images.append(caption_images_row)

    # Concatenate images and captions for each row
    final_rows = []
    for images_resized, caption_images_row in zip(images_resized_rows, caption_images):
        # Concatenate images in the current row
        final_images = [cv2.vconcat([img, caption]) for img, caption in zip(images_resized, caption_images_row)]
        final_row = cv2.hconcat(final_images)
        final_rows.append(final_row)

    # Concatenate all rows vertically
    final_image = cv2.vconcat(final_rows)

    return Image.fromarray(final_image)


def get_common_seed(): 

    # Get the current date and time
    current_date = datetime.datetime.now()

    # Convert to integer format YYYYMMDDHHMMSS
    date_as_integer = int(current_date.strftime("%d"))

    print("Date and Time in Integer Format:", date_as_integer)

    return date_as_integer 