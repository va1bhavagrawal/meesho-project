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
import pickle 


BLENDER_GRID_DIMS = {   
    "pickup_truck": 1.1, 
    "motorbike": 0.66, 
    "horse": 0.7, 
    "elephant": 1.1, 
    "lion": 0.85, 
    "jeep": 0.95, 
    "bus": 1.75,  
} 

OUTPUT_DIR = "2obj_renders_binned" 
OBJECTS_DIR = "obja_resized" 

NUM_AZIMUTH_BINS = 10  
NUM_DISTANCE_BINS = 10     
NUM_LIGHTS = 3    


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

    depsgraph = bpy.context.evaluated_depsgraph_get()

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


def reset_cameras(scene) -> None:
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


def add_plane():
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    backdrop = bpy.context.object
    mat_backdrop = bpy.data.materials.new(name="WhiteMaterial")
    mat_backdrop.diffuse_color = (0, 0, 0, 1)  # White
    backdrop.data.materials.append(mat_backdrop)


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


def set_lights():
    # Set the light
    # adding the flashlight 
    loc = VIEW_FRUSTUM["camera_pos"]
    bpy.ops.object.light_add(type='POINT', location=loc)  
    light = bpy.context.object 
    light.data.energy = 1000  
    light.data.use_shadow = False  

    for _ in range(NUM_LIGHTS): 
        loc = get_random_location() 
        loc[2] = random.randint(4, 6)  
        energy = random.randint(2000, 3000)  
        bpy.ops.object.light_add(type='POINT', location=loc)  
        light = bpy.context.object 
        light.data.energy = energy 
        light.data.use_shadow = True   


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


def render():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    context = bpy.context 
    scene = context.scene 
    # scene.eevee.use_gtao = True
    # scene.eevee.gtao_distance = 1.0
    # scene.eevee.gtao_factor = 1.0
    render = scene.render 
    # render.engine = "BLENDER_WORKBENCH" 
    render.engine = "BLENDER_EEVEE_NEXT"   
    bpy.context.scene.render.resolution_x = 1024  
    bpy.context.scene.render.resolution_y = 1024  
    bpy.context.scene.render.resolution_percentage = 100 
    # obj = get_object(f"obja_resized/pickup_truck.glb") 
    # obj.location = (-1, -15, 0)  

    reset_cameras(scene) 
    set_lights() 
    camera = scene.objects["Camera"] 
    camera.location = VIEW_FRUSTUM["camera_pos"]   
    direction = mathutils.Vector(VIEW_FRUSTUM["direction"])  
    bpy.context.scene.camera = camera 
    rot_quat = direction.to_track_quat("-Z", "Y") 
    camera.rotation_euler = rot_quat.to_euler() 
    camera.data.lens = VIEW_FRUSTUM["lens"]  

    objs = list(BLENDER_GRID_DIMS.keys())  
    objs = [f"{obj}.glb" for obj in objs] 
    obj_paths = sorted([osp.join(OBJECTS_DIR, obj) for obj in objs])  

    bpy.context.scene.render.resolution_x = 512 
    bpy.context.scene.render.resolution_y = 512 
    bpy.context.scene.render.resolution_percentage = 100 


    for i in range(6, 8, 1):  
        for j in range(len(obj_paths)):  
            obj1_path = obj_paths[i] 
            obj2_path = obj_paths[j]  

            obj1 = get_object(obj1_path)  
            obj2 = get_object(obj2_path) 
            add_plane()

            subject1 = osp.basename(obj1_path).replace(".glb", "").strip() 
            subject2 = osp.basename(obj2_path).replace(".glb", "").strip() 

            scale1 = BLENDER_GRID_DIMS[subject1] 
            scale2 = BLENDER_GRID_DIMS[subject2] 
            max_scale = max(scale1, scale2) 

            obj1.scale = (scale1 / max_scale, scale1 / max_scale, scale1 / max_scale)  
            obj2.scale = (scale2 / max_scale, scale2 / max_scale, scale2 / max_scale)  

            for distance_bin in range(NUM_DISTANCE_BINS): 
                for azimuth_bin in range(NUM_AZIMUTH_BINS): 
                    successful_render = False  
                    while not successful_render:  
                        near_x = VIEW_FRUSTUM["near"][0] 
                        far_x = VIEW_FRUSTUM["far"][0] 
                        delta_x = far_x - near_x  
                        x_range_min = near_x + (distance_bin / NUM_DISTANCE_BINS) * delta_x  
                        x_range_max = near_x + ((distance_bin+1) / NUM_DISTANCE_BINS) * delta_x  
                        x = random.uniform(x_range_min, x_range_max) 
                        h_range = get_h_range_at_distance(x)  
                        y = random.uniform(-h_range, h_range)  
                        obj1_loc = np.array([x, y, 0]) 
                        azimuth_range_min = (azimuth_bin / NUM_AZIMUTH_BINS) * 360 
                        azimuth_range_max = ((azimuth_bin+1) / NUM_AZIMUTH_BINS) * 360 
                        azimuth = random.uniform(int(azimuth_range_min), int(azimuth_range_max))  
                        obj1_azimuth = math.radians(azimuth)  
                        obj1.location = obj1_loc  
                        obj1.rotation_euler[2] = obj1_azimuth 

                        obj2_loc = get_random_location() 
                        obj2_azimuth = math.radians(random.uniform(0, 360)) 
                        obj2.location = obj2_loc  
                        obj2.rotation_euler[2] = obj2_azimuth 

                        save_path = osp.join(OUTPUT_DIR, f"{subject1}__{subject2}")  
                        os.makedirs(save_path, exist_ok=True) 
                        img_name = f"{obj1_loc[0]}_{obj1_loc[1]}_{obj1_loc[2]}_{obj1_azimuth}__{obj2_loc[0]}_{obj2_loc[1]}_{obj2_loc[2]}_{obj2_azimuth}__.png"  
                        pklname = img_name.replace("png", "pkl") 
                        pkl_path = osp.join(save_path, pklname) 
                        save_path = osp.join(save_path, img_name)  

                        bbox1 = get_object_2d_bbox(obj1, scene) 
                        bbox2 = get_object_2d_bbox(obj2, scene) 

                        iou = calculate_iou(bbox1, bbox2) 
                        if iou > 1e-3: 
                            continue 

                        successful_render = True 

                        # Set render settings
                        bpy.context.scene.render.filepath = "tmp.png"  
                        bpy.context.scene.render.image_settings.file_format = 'PNG' 
                        # Render the scene
                        bpy.ops.render.render(write_still=True) 

                        # successful_renders = successful_renders + 1 
                        shutil.copy("tmp.png", save_path)  

                        # img = cv2.imread(save_path) 
                        # cv2.rectangle(img, np.array((bbox1[0], bbox1[1])).astype(np.int32), np.array((bbox1[2], bbox1[3])).astype(np.int32), (255, 0, 0), 2) 
                        # cv2.rectangle(img, np.array((bbox2[0], bbox2[1])).astype(np.int32), np.array((bbox2[2], bbox2[3])).astype(np.int32), (0, 255, 0), 2) 
                        # cv2.imwrite(save_path, img) 

                        pkl_data = {} 
                        pkl_data["obj1"] = {} 
                        pkl_data["obj2"] = {} 

                        pkl_data["obj1"]["name"] = subject1 
                        pkl_data["obj1"]["a"] = obj1_azimuth 
                        pkl_data["obj1"]["location"] = np.array(obj1_loc)  
                        pkl_data["obj1"]["bbox"] = bbox1  

                        pkl_data["obj2"]["name"] = subject2 
                        pkl_data["obj2"]["a"] = obj2_azimuth 
                        pkl_data["obj2"]["location"] = np.array(obj2_loc) 
                        pkl_data["obj2"]["bbox"] = bbox2  

                        with open(pkl_path, "wb") as f: 
                            pickle.dump(pkl_data, f) 
                            

                        # change_position(obj1, (1.0, 0.0, 0.0)) 
                        # change_position(obj2, (-1.0, 0.0, 0.0)) 

            # delete_object(obj2) 
            # delete_object(obj2) 
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.select_by_type(type='MESH') 
            bpy.ops.object.delete()

        bpy.context.scene.render.filepath = "empty.png" 
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

    # Render the scene
    # bpy.context.scene.render.image_settings.file_format = 'PNG' 
    # bpy.context.scene.render.filepath = "tmp.jpg" 
    # bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    render()