import cv2
import random
import json
import argparse
import numpy as np
import torch
import os
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm
import math



def get_clip_score(image1, image2, model, processor):
    img1 = Image.open(image1).convert('RGB')
    img2 = Image.open(image2).convert('RGB')
    inputs1=processor(images=img1, return_tensors='pt', padding=True)
    inputs2=processor(images=img2, return_tensors='pt', padding=True)

     # 将输入数据移到 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    model.to(device)


    embed = []
    for inputs in [inputs1, inputs2]:
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            image_embeds = vision_outputs[1]
            image_embeds = model.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            embed.append(image_embeds)
    similarity = (100.0 * embed[0] @ embed[1].T).sum(dim=-1)
    return round(similarity.item(),4)


def process_folder(root_dir, model, processor):
    # 遍历根文件夹中的所有子文件夹
    
    for subfolder in os.listdir(root_dir):
        
        subfolder_path = os.path.join(root_dir, subfolder)
        dir=os.path.join('total_results',subfolder_path)
        if not os.path.isdir(subfolder_path):
            continue
        
        clothes_only_path = os.path.join(subfolder_path, 'clothes_only')
        person_wearing_clothes_path = os.path.join(subfolder_path, 'person_wearing_clothes')
        
        # 检查是否同时包含 clothes_only 和 person_wearing_clothes 文件夹
        if not (os.path.isdir(clothes_only_path) and os.path.isdir(person_wearing_clothes_path)):
            print(f"Skipping '{subfolder_path}' as it doesn't contain both required folders.")
            continue

        # 计算相似度并保存结果
        results = []
        for clothes_image in os.listdir(clothes_only_path):
            clothes_image_path = os.path.join(clothes_only_path, clothes_image)
            if not os.path.isfile(clothes_image_path):
                continue
            
            for person_image in os.listdir(person_wearing_clothes_path):
                person_image_path = os.path.join(person_wearing_clothes_path, person_image)
                if not os.path.isfile(person_image_path):
                    continue

                # 计算相似度
                similarity_score = get_clip_score(clothes_image_path, person_image_path, model, processor)
                #阈值设置
                if(similarity_score>90):
                    results.append({
                        'clothes_image': clothes_image,
                        'person_image': person_image,
                        'similarity_score': similarity_score
                    })
                    print(f"Similarity between '{clothes_image}' and '{person_image}' in '{subfolder}': {similarity_score}")

        # 保存结果到相应子文件夹中
        os.makedirs(dir, exist_ok=True)
        results_path = os.path.join(dir, 'clip_similarity_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved similarity results to '{results_path}'")



if __name__=='__main__':
    
    root_dir = "Samseg"
    
    # 初始化 CLIP 模型和处理器
    clip_path = "clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(clip_path)
    processor = CLIPProcessor.from_pretrained(clip_path)

    # 处理文件夹
    process_folder(root_dir, model, processor)
    '''
    image1="Samseg/4938275336663/person_wearing_clothes/O1CN01gMzl2m1tEVpgStVFG___3347915870.jpg"
    image2="Samseg/4938275336663/person_wearing_clothes/O1CN01yeB5Bm1tEVmUJIdXO___3347915870.jpg"
    image3="Samseg/4938275336663/person_wearing_clothes/O1CN01ltMFE51tEVmLchcBL___3347915870.jpg"
    print(get_clip_score(image1,image2,model,processor))
    print(get_clip_score(image2,image3,model,processor))
    '''



