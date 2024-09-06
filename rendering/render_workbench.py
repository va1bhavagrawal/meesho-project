import bpy 
import bpy_extras 
import numpy as np 
import os 
import os.path as osp 
import shutil 
import sys 
import math 
import mathutils 
import random 
import cv2 


BLENDER_GRID_DIMS = {   
    "pickup_truck": 1.1, 
    "motorbike": 0.66, 
    "horse": 0.7, 
    "elephant": 1.1, 
    "lion": 0.85, 
    "jeep": 0.95, 
    "bus": 2.0,  
} 

OBJECTS_DIR = "obja_resized" 
WORKBENCH_DIR = "2obj_renders"
OUTPUT_DIR = "2obj_renders_workbench" 

context = bpy.context 
scene = context.scene 
render = scene.render 
render.engine = "BLENDER_WORKBENCH"   
bpy.context.scene.render.resolution_x = 512 
bpy.context.scene.render.resolution_y = 512 
bpy.context.scene.render.resolution_percentage = 100


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: Each box is defined by a tuple (x1, y1, x2, y2)
                where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    float: IoU value
    """
    # Unpack coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Determine the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Compute the area of intersection rectangle
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def get_object_2d_bbox(empty_obj, scene):
    """
    Get the 2D bounding box coordinates of an object in the rendered image.
    
    Args:
        empty_obj (bpy.types.Object): The empty object containing the child mesh objects.
        scene (bpy.types.Scene): The current scene.
        
    Returns:
        tuple: A tuple containing the 2D bounding box coordinates in pixel space
              in the format (min_x, min_y, max_x, max_y).
    """
    # Get the render settings
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    
    # Initialize the bounding box coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # Iterate through the child mesh objects
    for obj in empty_obj.children:
        if obj.type == 'MESH':
            # Get the bounding box coordinates in world space
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            # Transform the bounding box corners to camera space
            for corner in bbox_corners:
                corner_2d = bpy_extras.object_utils.world_to_camera_view(scene, scene.camera, corner)
                
                # Scale the coordinates to pixel space
                x = corner_2d.x * res_x
                y = (1 - corner_2d.y) * res_y  # Flip Y since Blender renders from bottom to top
                
                # Update the bounding box coordinates
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    
    # Return the 2D bounding box coordinates in pixel space
    return (int(min_x), int(min_y), int(max_x), int(max_y))


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object  
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera

reset_cameras() 


def get_object(obj_path): 
    # Load the first GLB object
    # Create an empty object to use as a parent
    bpy.ops.object.empty_add(type="PLAIN_AXES")  
    empty_object = bpy.context.object
    before_objs = set(bpy.data.objects) 
    bpy.ops.import_scene.gltf(filepath=obj_path)
    after_objs = set(bpy.data.objects) 
    diff_objs = after_objs - before_objs 
    for obj in diff_objs: 
        obj.parent = empty_object 
        world_matrix = obj.matrix_world 
        obj.matrix_world = world_matrix 
    return empty_object  


# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()


def change_position(obj: list, delta_pos): 
    # for obj in obj_list: 
    #     obj.location = np.array(obj.location) + np.array(delta_pos)  
    obj.location = np.array(obj.location) + np.array(delta_pos) 


def delete_object(obj): 
    bpy.context.view_layer.objects.active = obj 
    obj.select_set(True) 
    bpy.ops.object.delete() 


# for each tuple, the first element is the coordinate in the direction in which the camera is facing  
VIEW_FRUSTUM = {
    "near": (-2.7, 0.50),  
    "far": (-7.0, 1.5), 
    "lens": 50,   
    "camera_pos": (1.5, 0.0, 1.75),  
    "direction": (-1, 0, -0.15), 
}

def get_h_range_at_distance(distance): 
    ratio_value = (distance - VIEW_FRUSTUM["near"][0]) / (VIEW_FRUSTUM["far"][0] - VIEW_FRUSTUM["near"][0])  
    abs_h_value = VIEW_FRUSTUM["near"][1] + ratio_value * (VIEW_FRUSTUM["far"][1] - VIEW_FRUSTUM["near"][1])  
    return abs_h_value 

def get_random_location(): 
    x = random.uniform(VIEW_FRUSTUM["far"][0], VIEW_FRUSTUM["near"][0]) 
    h_range = get_h_range_at_distance(x) 
    y = random.uniform(-h_range, h_range) 
    return np.array([x, y, 0]) 


camera = scene.objects["Camera"] 
camera.location = VIEW_FRUSTUM["camera_pos"]   
# direction = -camera.location 
direction = mathutils.Vector(VIEW_FRUSTUM["direction"])  
# camera_object.rotation_euler = (1.1, 0, 0)  # Adjust rotation as needed
bpy.context.scene.camera = camera 
rot_quat = direction.to_track_quat("-Z", "Y") 
camera.rotation_euler = rot_quat.to_euler() 
camera.data.lens = VIEW_FRUSTUM["lens"]  

# Render the scene from different camera views
objs = os.listdir(osp.join(OBJECTS_DIR)) 
# removing giraffe since it was too tall and causing issues w.r.t. rendering 
objs = [obj for obj in objs if obj.find("giraffe") == -1 and obj.find("cat") == -1]   
obj_paths = sorted([osp.join(OBJECTS_DIR, obj) for obj in objs])  

for subject_pair in os.listdir(WORKBENCH_DIR): 
    subject1 = subject_pair.split("__")[0]  
    subject2 = subject_pair.split("__")[1]   
    obj1_path = osp.join(OBJECTS_DIR, f"{subject1}.glb")  
    obj2_path = osp.join(OBJECTS_DIR, f"{subject2}.glb")  
    
    obj1 = get_object(obj1_path) 
    obj2 = get_object(obj2_path) 

    subject_pair_path = osp.join(WORKBENCH_DIR, subject_pair) 
    for img_name in os.listdir(subject_pair_path): 
        subject1_coords, subject2_coords, _ = img_name.split("__") 
        x1, y1, z1, a1 = subject1_coords.split("_") 
        x2, y2, z2, a2 = subject2_coords.split("_") 
        x1 = float(x1) 
        y1 = float(y1) 
        z1 = float(z1) 
        a1 = float(a1) 
        x2 = float(x2) 
        y2 = float(y2) 
        z2 = float(z2) 
        a2 = float(a2) 

        scale1 = BLENDER_GRID_DIMS[subject1] 
        scale2 = BLENDER_GRID_DIMS[subject2] 
        max_scale = max(scale1, scale2) 

        obj1.scale = (scale1 / max_scale, scale1 / max_scale, scale1 / max_scale)  
        obj2.scale = (scale2 / max_scale, scale2 / max_scale, scale2 / max_scale)  

        successful_renders = 0 

        # obj1_loc = get_random_location() 
        # obj2_loc = get_random_location() 
        obj1_loc = np.array([x1, y1, z1]) 
        obj2_loc = np.array([x2, y2, z2])  

        # print(f"{obj1_loc = }")
        # print(f"{obj2_loc = }")

        # obj1_azimuth = math.radians(90) 
        # obj2_azimuth = math.radians(90)  

        obj1.location = obj1_loc  
        obj2.location = obj2_loc  
        obj1.rotation_euler[2] = a1 
        obj2.rotation_euler[2] = a2  

        save_path = osp.join(OUTPUT_DIR, subject_pair)  
        os.makedirs(save_path, exist_ok=True) 
        # save_path = osp.join(save_path, f"{obj1_loc[0]}_{obj1_loc[1]}_{obj1_loc[2]}_{obj1_azimuth}__{obj2_loc[0]}_{obj2_loc[1]}_{obj2_loc[2]}_{obj2_azimuth}__.png") 
        save_path = osp.join(save_path, f"{x1}_{y1}_{z1}_{a1}__{x2}_{y2}_{z2}_{a2}__.jpg")  

        # Set render settings
        bpy.context.scene.render.filepath = save_path  
        bpy.context.scene.render.image_settings.file_format = 'JPEG' 
        
        # Render the scene
        bpy.ops.render.render(write_still=True) 

        # get the bbox  
        # bbox1 = get_object_2d_bbox(obj1, scene) 
        # bbox2 = get_object_2d_bbox(obj2, scene) 
        # print(f"{bbox1 = }")
        # print(f"{bbox2 = }")

        # iou = calculate_iou(bbox1, bbox2) 
        # if iou > 0.0:  
        #     continue 

        # successful_renders = successful_renders + 1 
        # shutil.copy("tmp.png", save_path)  

        # img = cv2.imread(save_path) 
        # bbox = get_object_2d_bbox(obj2, scene) 
        # print(f"{bbox = }") 
        # cv2.rectangle(img, np.array((bbox1[0], bbox1[1])).astype(np.int32), np.array((bbox1[2], bbox1[3])).astype(np.int32), (255, 0, 0), 2) 
        # cv2.rectangle(img, np.array((bbox2[0], bbox2[1])).astype(np.int32), np.array((bbox2[2], bbox2[3])).astype(np.int32), (0, 255, 0), 2) 
        # cv2.imwrite(save_path, img) 

        # change_position(obj1, (1.0, 0.0, 0.0)) 
        # change_position(obj2, (-1.0, 0.0, 0.0)) 

    # delete_object(obj2) 
    # delete_object(obj2) 
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH') 
    bpy.ops.object.delete()

    bpy.context.scene.render.filepath = "empty.png" 
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # Render the scene
    bpy.ops.render.render(write_still=True)
    # Optionally, remove the camera after rendering
    # bpy.data.objects.remove(camera_object)