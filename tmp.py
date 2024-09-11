import os 
import os.path as osp 
import shutil 

SRC_DIR = "../rendering/2obj_renders_large/" 
DEST_DIR = "training_data_2subjects_0409/ref_imgs_2subjects/" 

for subjects_string in os.listdir(SRC_DIR): 
    src_dir = osp.join(SRC_DIR, subjects_string) 
    dst_dir = osp.join(DEST_DIR, subjects_string) 

    assert osp.exists(src_dir) 
    assert osp.exists(dst_dir) 

    src_paths = os.listdir(src_dir) 
    # dst_paths = os.listdir(dst_dir) 
    src_paths = [osp.join(src_dir, src_path) for src_path in src_paths] 
    # dst_paths = [osp.join(dst_dir, dst_path) for dst_path in dst_paths] 

    if len(src_paths) != 1922: 
        print(f"{subjects_string} could not be generated completely!") 

    # print(f"{len(src_paths) = }")
    # print(f"{len(dst_paths) = }")

    for src_path in src_paths: 
        basename = osp.basename(src_path) 
        dst_path = osp.join(DEST_DIR, subjects_string, basename) 
        # print(f"{src_path = }")
        # print(f"{dst_path = }")
        # import sys 
        # sys.exit(0) 

        shutil.copy(src_path, dst_path) 