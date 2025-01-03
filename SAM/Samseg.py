import pickle
from segment_anything import SamPredictor, sam_model_registry
import segment_anything
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

model_type = "default"  # 例如 'vit_b', 'vit_l' 等
checkpoint_path = "H:\\university\\machinelearning\\Virtual_Clothing\\ckpt\\sam_vit_h_4b8939.pth"  # 替换为实际的checkpoint路径
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

#"H:\university\machinelearning\Virtual_Clothing\Virtual-Try-on-Clothing\params.pkl"
# bbox_dict_person = pickle.load(open("H:\\university\\machinelearning\\Virtual_Clothing\\Virtual-Try-on-Clothing\\params_person.pkl", "rb"))
# bbox_dict_clothes = pickle.load(open("H:\\university\\machinelearning\\Virtual_Clothing\\Virtual-Try-on-Clothing\\params_clothes.pkl", "rb"))
# bbox_dict = {**bbox_dict_person, **bbox_dict_clothes}
# key: Label , Center X, Center Y, Width, Height
#---- washed
#    ---- graph num
#        ---- clothes_only
#        ---- person_wearing_clothes

file_root_person = "J:\\university\\machinelearning\\openpose"
file_root_clothes = "J:\\university\\machinelearning\\washed"
output_root = "J:\\university\\machinelearning\\SAM"
if not os.path.exists(output_root):
    os.mkdir(output_root)

# box_root = file_root_clothes
# bbox_dict = {}
# for file in os.listdir(box_root):
#     dict_path = os.path.join(box_root, file, "params.pkl")
#     dict = pickle.load(open(dict_path, "rb"))
#     bbox_dict = {**bbox_dict, **dict}
bbox_dict = pickle.load(open("J:\\university\\machinelearning\\final_dict.pkl", "rb"))
class_dict = pickle.load(open("J:\\university\\machinelearning\\class_dict.pkl", "rb"))
pose_dict = pickle.load(open("J:\\university\\machinelearning\\pose_dict.pkl", "rb"))

def segandsave(file, file_name, class_type, pose, clothes_type):
    #file: 文件路径
    #file_name: 文件名
    #class_type: 类别 person or clothes
    #mask_name: 保存的文件名
    #clothes_type: 衣服类型 例如polo衫
    for name in os.listdir(file):
        # save_path = os.path.join("Samseg", file_name, class_type)
        if class_type == 'clothes':
            save_path = os.path.join(output_root, clothes_type, file_name, class_type)
        elif class_type == 'person':
            save_path = os.path.join(output_root, clothes_type, file_name, class_type, pose)
        else:
            raise ValueError("class type error")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img = Image.open(os.path.join(file, name)).convert("RGB")
        img = np.array(img)
        predictor.set_image(img)
        img_key = file_name + "_" + name
        if img_key not in bbox_dict:
            print("No bbox for ", img_key)
            return
        print('find bbox for ', img_key)
        box = [
            bbox_dict[img_key]["Center X"] - bbox_dict[img_key]["Width"] // 2,
            bbox_dict[img_key]["Center Y"] - bbox_dict[img_key]["Height"] // 2,
            bbox_dict[img_key]["Center X"] + bbox_dict[img_key]["Width"] // 2,
            bbox_dict[img_key]["Center Y"] + bbox_dict[img_key]["Height"] // 2
        ]
        box = np.array(box)
        
        center_point = (bbox_dict[file_name+"_"+name]["Center X"], bbox_dict[file_name+"_"+name]["Center Y"])
        center_point = np.array(center_point)
        center_point = center_point.reshape(1, 2)
        
        masks, _, _ = predictor.predict(point_coords=center_point, point_labels=np.array([1,]), box=box)
        masks = masks.transpose(1, 2, 0)
        masks = np.sum(masks, axis=-1)
        masks = masks >= 2
        masks = np.expand_dims(masks, axis=-1)
        masked_img = img * masks
        masked_img = Image.fromarray(masked_img.astype(np.uint8))
        masked_img.save(os.path.join(save_path, name))


#use tqdm
for file_washed in tqdm.tqdm(os.listdir(file_root_clothes), desc="clothes"):
    clothes_type = file_washed
    file_washed = os.path.join(file_root_clothes, file_washed)
    for graph_file in tqdm.tqdm(os.listdir(file_washed), desc="graph"):
        if os.path.exists(os.path.join(file_washed, graph_file, "clothes_only")):
            clothes_file = os.path.join(file_washed, graph_file, "clothes_only")
            segandsave(clothes_file, graph_file, "clothes", None, clothes_type)

for file_washed in tqdm.tqdm(os.listdir(file_root_person)):
    clothes_type = file_washed
    file_washed = os.path.join(file_root_person, file_washed)
    for graph_file in tqdm.tqdm(os.listdir(file_washed), desc="graph"):
        for pose in os.listdir(os.path.join(file_washed, graph_file, 'person')):
            if os.path.exists(os.path.join(file_washed, graph_file, 'person', pose)):
                person_file = os.path.join(file_washed, graph_file, 'person', pose)
                segandsave(person_file, graph_file, "person", pose, clothes_type)








    
    
    
        
        
        