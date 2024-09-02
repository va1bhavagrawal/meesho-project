import os 
import os.path as osp 
from PIL import Image 

REF_IMGS_DIR = osp.join(f"rendering", "2obj_renders_binned") 
count_removed = 0 
total_count = 0 
DUMP_DIR = f"dump" 
import shutil 
if osp.exists(DUMP_DIR):  
    shutil.rmtree(DUMP_DIR) 
os.mkdir(DUMP_DIR) 

for subject_pair_ in os.listdir(REF_IMGS_DIR): 
    subject_pair_path = osp.join(REF_IMGS_DIR, subject_pair_) 
    for img_name in os.listdir(subject_pair_path): 
        if img_name.find(f"png") == -1: 
            continue 
        total_count += 1 
        img_path = osp.join(subject_pair_path, img_name) 
        try: 
            img = Image.open(img_path) 
            img.save(osp.join(DUMP_DIR, img_name)) 
        except: 
            print(f"removing {img_path}")
            pkl_path = img_path.replace("png", "pkl") 
            count_removed += 1
            assert osp.exists(pkl_path) 
            assert osp.exists(img_path) 
            # os.remove(pkl_path) 
            # os.remove(img_path) 
print(f"{count_removed = }")
print(f"{total_count = }") 