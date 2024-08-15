import numpy as np
import os 
import os.path as osp 

import torch 
import torch.nn as nn 

from tqdm import tqdm 

import math 

from continuous_word_mlp import continuous_word_mlp, MergedEmbedding, AppearanceEmbeddings  

from transformers import CLIPTextModel  

import copy 
# Example: Replace this with your actual embeddings and labels
# embeddings = np.random.rand(100, 50)  # 100 samples with 50 dimensions
# labels = np.random.randint(0, 5, 100)  # 5 different labels

TOKEN2ID = {
    "sks": 48136, 
    "bnha": 49336,  
    "pickup truck": 4629, # using the token for "truck" instead  
    "bus": 2840, 
    "cat": 2368, 
    "giraffe": 22826, 
    "horse": 4558,
    "lion": 5567,  
    "elephant": 10299,   
    "jeep": 11286,  
    "motorbike": 33341,  
    "bicycle": 11652, 
    "tractor": 14607,  
    "truck": 4629,  
    "zebra": 22548,  
    "sedan": 24237, 
    "motocross": 34562,  
    "boat": 4440,   
    "ship": 1158,  
    "plane":5363,  
    "helicopter": 11956,   
    "shoe": 7342,  
    "bird": 3329,  
    "sparrow": 22888,  
    "suitcase": 27792,  
    "chair": 4269,  
    "dolphin": 16464, 
    "fish": 2759, 
    "shark": 7980, 
    "man": 786, 
    "dog": 1929, 
}


PERPLEXITY = 30  

WHICH_MODEL = "__poseonly_nosubject_skip"  
WHICH_STEP = 100000  

def get_embeddings(subjects, pose_types, appearance_types, azimuths, mlp, merger, text_encoder, bnha_embeds=None): 
    assert len(subjects) == len(pose_types) == len(appearance_types) 
    embeddings = [] 
    for subject, pose_type, appearance_type, azimuth in tqdm(zip(subjects, pose_types, appearance_types, azimuths)):   
        if pose_type == "0": 
            mlp_emb = torch.zeros(1024).to(0) 
        elif pose_type == "a":  
            sincos = torch.Tensor([torch.sin(2 * torch.pi * torch.tensor(azimuth)), torch.cos(2 * torch.pi * torch.tensor(azimuth))]).to(0)  
            mlp_emb = mlp(sincos.unsqueeze(0)).squeeze()  
        else: 
            assert False 

        if appearance_type == "zero": 
            bnha_emb = torch.zeros(1024).to(0) 
        elif appearance_type == "learnt":  
            assert bnha_embeds is not None 
            bnha_emb = getattr(bnha_embeds, subject) 
        elif appearance_type == "class": 
            bnha_emb = text_encoder.get_input_embeddings().weight[TOKEN2ID[subject]]  
        else: 
            assert False, f"{appearance_type = }"

        if pose_type == "0" and appearance_type == "zero": 
            merged_emb = text_encoder.get_input_embeddings().weight[TOKEN2ID[subject]] 
        else: 
            merged_emb = merger(mlp_emb, bnha_emb)  

        embeddings.append(merged_emb) 
    
    return torch.stack(embeddings)  


with torch.no_grad(): 
    ckpt_path = f"../ckpts/multiobject/{WHICH_MODEL}/training_state_{WHICH_STEP}.pth"  
    basename_ckpt = osp.basename(ckpt_path) 
    lora_path = ckpt_path.replace(basename_ckpt, f"lora_weight_{WHICH_STEP}.safetensors") 

    training_state = torch.load(ckpt_path) 

    merger = MergedEmbedding().to(0) 
    merger.load_state_dict(training_state["merger"]["model"])  

    mlp = continuous_word_mlp(2, 1024).to(0)  
    mlp.load_state_dict(training_state["contword"]["model"]) 

    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder").to(0)  

    unique_subjects = ["bus", "elephant", "lion", "horse", "cat", "pickup truck", "bus", "motorbike", "tractor", "sedan", "bicycle", "motocross", "boat", "dog"] 

    unique_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
          '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000']

    subject2color = {} 
    for color, subject in zip(unique_colors, unique_subjects): 
        subject2color[subject] = color 

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches 


    ######################################################### pose+class embedding ##########################3 
    pose_class = {}  
    subjects = [] 
    pose_types = [] 
    appearance_types = [] 
    azimuths = [] 
    for subject in unique_subjects: 
        for azimuth in np.linspace(0, 2 * math.pi, 180):   
            for pose_type in ["a"]: 
                for appearance_type in ["class"]: 
                    subjects.append(subject) 
                    azimuths.append(azimuth) 
                    pose_types.append(pose_type) 
                    appearance_types.append(appearance_type) 
                    
    print(f"making the pose+class embeddings...")
    embeddings = get_embeddings(subjects, pose_types, appearance_types, azimuths, mlp, merger, text_encoder)  
    embeddings = embeddings.cpu().numpy() 
    
    pose_class["embeddings"] = copy.deepcopy(embeddings)  
    pose_class["subjects"] = copy.deepcopy(subjects)  
    pose_class["name"] = "pose_class"
    pose_class["marker"] = "o" 


    ############################################### class only ###############################
    class_only = {} 
    subjects = [] 
    pose_types = [] 
    appearance_types = [] 
    azimuths = [] 
    # zero+zero = class embeddings 
    for subject in unique_subjects: 
        for azimuth in [0]:  
            for pose_type in ["0"]: 
                for appearance_type in ["zero"]: 
                    subjects.append(subject) 
                    azimuths.append(azimuth) 
                    pose_types.append(pose_type) 
                    appearance_types.append(appearance_type) 

    print(f"making the class only embeddings...")
    embeddings = get_embeddings(subjects, pose_types, appearance_types, azimuths, mlp, merger, text_encoder)  
    embeddings = embeddings.cpu().numpy() 

    assert len(embeddings) == len(unique_subjects) 
    
    class_only["embeddings"] = copy.deepcopy(embeddings)  
    class_only["subjects"] = copy.deepcopy(subjects) 
    class_only["name"] = "class_only" 
    class_only["marker"] = "x" 


    ############################################### class+0 only ###############################
    class_zero = {} 
    subjects = [] 
    pose_types = [] 
    appearance_types = [] 
    azimuths = [] 
    # zero+zero = class embeddings 
    for subject in unique_subjects: 
        for azimuth in [0]:  
            for pose_type in ["0"]: 
                for appearance_type in ["class"]: 
                    subjects.append(subject) 
                    azimuths.append(azimuth) 
                    pose_types.append(pose_type) 
                    appearance_types.append(appearance_type) 

    print(f"making the class only embeddings...")
    embeddings = get_embeddings(subjects, pose_types, appearance_types, azimuths, mlp, merger, text_encoder)  
    embeddings = embeddings.cpu().numpy() 

    assert len(embeddings) == len(unique_subjects) 
    
    class_zero["embeddings"] = copy.deepcopy(embeddings)  
    class_zero["subjects"] = copy.deepcopy(subjects) 
    class_zero["name"] = "class_zero" 
    class_zero["marker"] = "^"  


    ############################################### 0+pose only ###############################
    # zero_pose = {} 
    # subjects = [] 
    # pose_types = [] 
    # appearance_types = [] 
    # azimuths = [] 
    # # zero+zero = class embeddings 
    # for subject in unique_subjects: 
    #     for azimuth in [0]:  
    #         for pose_type in ["a"]: 
    #             for appearance_type in ["zero"]: 
    #                 subjects.append(subject) 
    #                 azimuths.append(azimuth) 
    #                 pose_types.append(pose_type) 
    #                 appearance_types.append(appearance_type) 

    # print(f"making the class only embeddings...")
    # embeddings = get_embeddings(subjects, pose_types, appearance_types, azimuths, mlp, merger, text_encoder)  
    # embeddings = embeddings.cpu().numpy() 

    # assert len(embeddings) == len(unique_subjects) 
    
    # zero_pose["embeddings"] = copy.deepcopy(embeddings)  
    # zero_pose["subjects"] = copy.deepcopy(subjects) 
    # zero_pose["name"] = "zero_pose" 
    # zero_pose["marker"] = "+"  


    ############################# combining all of them #########################################  
    collections = [pose_class, class_only, class_zero]  

    from sklearn.manifold import TSNE
    all_embeddings = [] 
    for collection in collections: 
        all_embeddings.append(collection["embeddings"]) 
    all_embeddings = np.concatenate(all_embeddings, 0) 
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)


    # trying to create another legend for the type of embedding 
    # handles = [mpatches.Patch(marker=collection["marker"], label=collection["name"]) for collection in collections]  
    # plt.legend(handles=handles, title="embedding_type", bbox_to_anchor=(1, 1.5), loc='right') 


    n_embeddings = 0 
    plt.figure(figsize=(15, 15))
    scatters = [] 
    for collection in collections:  
        collection_reduced_embeddings = reduced_embeddings[n_embeddings : n_embeddings + len(collection["embeddings"])] 
        colors = [] 
        for subject in collection["subjects"]: 
            colors.append(subject2color[subject]) 
        # if collection["name"] == pose_class: 
        #     scatter = plt.scatter(collection_reduced_embeddings[:, 0], collection_reduced_embeddings[:, 1], c=colors, alpha=0.7) 
        # else:  
        #     scatter = plt.scatter(collection_reduced_embeddings[:, 0], collection_reduced_embeddings[:, 1], c=colors, alpha=0.7, marker="x") 
        scatter = plt.scatter(collection_reduced_embeddings[:, 0], collection_reduced_embeddings[:, 1], c=colors, alpha=0.7, marker=collection["marker"], label=collection["name"])    
        scatters.append(scatter) 
        n_embeddings += len(collection_reduced_embeddings) 


    # Create a legend manually 
    handles = [mpatches.Patch(color=subject2color[subject], label=subject) for subject in unique_subjects] 
    plt.legend(handles=handles, title="Subjects", bbox_to_anchor=(1, 1), loc='upper left') 

    # plt.legend(handles=scatters) 

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    # plt.colorbar(scatter, label='Labels')
    plt.show()
    plt.savefig(f"plot_{PERPLEXITY}.jpg") 