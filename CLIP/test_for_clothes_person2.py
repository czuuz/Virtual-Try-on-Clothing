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

def get_clip_score_single(image1,  model, processor):
    img1 = Image.open(image1).convert('RGB')
    
    inputs1=processor(images=img1, return_tensors='pt', padding=True)
    

     # 将输入数据移到 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    model.to(device)

    embed = []
    
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs1)
        image_embeds = vision_outputs[1]
        image_embeds = model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        embed.append(image_embeds)
    return embed[0]


def process_folder(root_dir, model, processor):
    # 遍历根文件夹中的所有子文件夹
    
    for subfolder in os.listdir(root_dir):
        
        subfolder_path = os.path.join(root_dir, subfolder)
        dir=os.path.join('total_results_clothes60',subfolder_path)
        if not os.path.isdir(subfolder_path):
            continue
        
        clothes_only_path = os.path.join(subfolder_path, 'clothes')  #clothes_only
        person_wearing_clothes_path = os.path.join(subfolder_path, 'person') #person_wearing_clothes
        
        # 检查是否同时包含 clothes_only 和 person_wearing_clothes 文件夹
        if not (os.path.isdir(clothes_only_path) and os.path.isdir(person_wearing_clothes_path)):
            print(f"Skipping '{subfolder_path}' as it doesn't contain both required folders.")
            continue
        Embed={}
        hist={}
        for clothes_image in os.listdir(clothes_only_path):
            clothes_image_path = os.path.join(clothes_only_path, clothes_image)
            #print(clothes_image_path)
            if not os.path.isfile(clothes_image_path):
                continue
            Embed[clothes_image] = get_clip_score_single(clothes_image_path,  model, processor)
            hist[clothes_image] = get_color_histogram_with_cuda(clothes_image_path)
        
        for person_wearing_clothes_pose in os.listdir(person_wearing_clothes_path):
            person_wearing_clothes_path2=os.path.join(person_wearing_clothes_path,person_wearing_clothes_pose)
            if not os.path.isdir(person_wearing_clothes_path2):
                    continue
                #print(666)
            for person_image in os.listdir(person_wearing_clothes_path2):
                    
                person_image_path = os.path.join(person_wearing_clothes_path2, person_image)
                    #print(777)
                if not os.path.isfile(person_image_path):
                    continue
                Embed[person_image] = get_clip_score_single(person_image_path,  model, processor)
                hist[person_image] = get_color_histogram_with_cuda(person_image_path)


        # 计算相似度并保存结果
        results = []
        for clothes_image in os.listdir(clothes_only_path):
            clothes_image_path = os.path.join(clothes_only_path, clothes_image)
            #print(clothes_image_path)
            if not os.path.isfile(clothes_image_path):
                continue
            list1= [clothes_image]
            list2= []
            
            for person_wearing_clothes_pose in os.listdir(person_wearing_clothes_path):
                person_wearing_clothes_path2=os.path.join(person_wearing_clothes_path,person_wearing_clothes_pose)
                if not os.path.isdir(person_wearing_clothes_path2):
                        continue
                #print(666)
                for person_image in os.listdir(person_wearing_clothes_path2):
                    
                    person_image_path = os.path.join(person_wearing_clothes_path2, person_image)
                    #print(777)
                    if not os.path.isfile(person_image_path):
                        continue
                    # 计算相似度
                    ####
                    ####
                    # 将颜色直方图从 NumPy 转换为 PyTorch 张量并移到 GPU
                    hist1_r, hist1_g, hist1_b = torch.tensor(hist[clothes_image][0], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist[clothes_image][1], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist[clothes_image][2], dtype=torch.float32).cuda()
                    hist2_r, hist2_g, hist2_b = torch.tensor(hist[person_image][0], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist[person_image][1], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist[person_image][2], dtype=torch.float32).cuda()

                    # 计算 R、G、B 通道的相似度
                    sim_r = cosine_similarity(hist1_r, hist2_r)
                    sim_g = cosine_similarity(hist1_g, hist2_g)
                    sim_b = cosine_similarity(hist1_b, hist2_b)

                    color_similarity =(sim_r + sim_g + sim_b) / 3

                    similarity = (100.0 * Embed[clothes_image] @ Embed[person_image].T).sum(dim=-1)
                    clip_similarity =  round(similarity.item(),4)

                    # 计算最终相似度
                    similarity_score = 0.5 * clip_similarity + 0.5 * (color_similarity * 100) 
                    #阈值设置
                    print(similarity_score.cpu().item())
                    #print(f"Similarity between '{clothes_image}' and '{person_image}' in '{subfolder}': {similarity_score}")
                    if(similarity_score.cpu().item()>60):  #86.5
                        list2.append(person_image)
                        
                        print(f"Similarity between '{clothes_image}' and '{person_image}' in '{subfolder}': {similarity_score}")
            if(len(list2)==0):
                continue
            results.append({
                            'clothes_image': list1,
                            'person_image': list2,
                            #'similarity_score': similarity_score
                        })
        if(len(results)==0):
            continue
        # 保存结果到相应子文件夹中
        os.makedirs(dir, exist_ok=True)
        results_path = os.path.join(dir, 'clothes_similarity_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved similarity results to '{results_path}'")
'''
def get_color_histogram(image_path):
    """
    计算图像的颜色直方图
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像分成 R、G、B 三个通道
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
    # 归一化
    hist_r = hist_r / hist_r.sum()
    hist_g = hist_g / hist_g.sum()
    hist_b = hist_b / hist_b.sum()
    return hist_r, hist_g, hist_b
'''

def get_color_histogram(image_path):
    '''
    计算图像的颜色直方图，排除像素值为零的区域
    
    img = cv2.imread(image_path)
    #img=   Image.open(image_path).convert('RGB')
    if img is None:
        print(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 生成掩码，只保留非零像素
    mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
    
    # 分别计算 R、G、B 通道的直方图，使用 mask 排除零值
    hist_r = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], mask, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], mask, [256], [0, 256])
    
    # 归一化
    hist_r = hist_r / hist_r.sum() if hist_r.sum() > 0 else hist_r
    hist_g = hist_g / hist_g.sum() if hist_g.sum() > 0 else hist_g
    hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b
    
    return hist_r, hist_g, hist_b
    '''
    """
    计算图像的颜色直方图，排除像素值为零的区域
    """
    # 使用 PIL 打开图像
    img = Image.open(image_path)
    img = img.convert('RGB')  # 转换为 RGB 模式

    # 将图像转换为 NumPy 数组
    img_array = np.array(img)

    # 创建掩码，排除像素值为 (0, 0, 0) 的区域
    mask = np.all(img_array != [0, 0, 0], axis=-1)

    # 使用掩码过滤掉零值区域
    img_array = img_array[mask]

    # 计算 R、G、B 通道的直方图
    hist_r, _ = np.histogram(img_array[:, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(img_array[:, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(img_array[:, 2], bins=256, range=(0, 256))

    # 归一化
    hist_r = hist_r / hist_r.sum() if hist_r.sum() > 0 else hist_r
    hist_g = hist_g / hist_g.sum() if hist_g.sum() > 0 else hist_g
    hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b

    return hist_r, hist_g, hist_b

def get_color_histogram_with_cuda(image_path):
    """
    使用 PyTorch 库计算图像的颜色直方图，排除像素值为零的区域
    """
    # 使用 PIL 打开图像
    img = Image.open(image_path)
    img = img.convert('RGB')  # 转换为 RGB 模式

    # 将图像转换为 NumPy 数组并转为 PyTorch 张量
    img_array = np.array(img)
    img_tensor = torch.tensor(img_array, dtype=torch.uint8).cuda()  # 将 NumPy 数组转换为 PyTorch 张量并移到 GPU

    # 创建掩码，排除像素值为 (0, 0, 0) 的区域
    mask = torch.all(img_tensor != torch.tensor([0, 0, 0], dtype=torch.uint8).cuda(), dim=-1)

    # 使用掩码过滤掉零值区域
    img_tensor = img_tensor[mask]

    # 计算 R、G、B 通道的直方图
    hist_r = torch.histc(img_tensor[:, 0].float(), bins=256, min=0, max=255)
    hist_g = torch.histc(img_tensor[:, 1].float(), bins=256, min=0, max=255)
    hist_b = torch.histc(img_tensor[:, 2].float(), bins=256, min=0, max=255)

    # 归一化
    hist_r = hist_r / hist_r.sum() if hist_r.sum() > 0 else hist_r
    hist_g = hist_g / hist_g.sum() if hist_g.sum() > 0 else hist_g
    hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b

    return hist_r, hist_g, hist_b


'''

def calculate_color_similarity(hist1, hist2):
    """
    计算两组颜色直方图的相似性，使用余弦相似度
    """
    def cosine_similarity(h1, h2):
        return np.dot(h1.T, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
    
    sim_r = cosine_similarity(hist1[0], hist2[0])
    sim_g = cosine_similarity(hist1[1], hist2[1])
    sim_b = cosine_similarity(hist1[2], hist2[2])
    
    # 平均颜色相似度
    return (sim_r + sim_g + sim_b) / 3
'''
def cosine_similarity(h1, h2):
    return torch.matmul(h1.T, h2) / (torch.norm(h1) * torch.norm(h2))
def calculate_color_similarity_with_cuda(hist1, hist2):
    """
    使用 PyTorch 库计算两组颜色直方图的相似性，使用余弦相似度
    """

    # 将颜色直方图从 NumPy 转换为 PyTorch 张量并移到 GPU
    hist1_r, hist1_g, hist1_b = torch.tensor(hist1[0], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist1[1], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist1[2], dtype=torch.float32).cuda()
    hist2_r, hist2_g, hist2_b = torch.tensor(hist2[0], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist2[1], dtype=torch.float32).cuda(), \
                                 torch.tensor(hist2[2], dtype=torch.float32).cuda()

    # 计算 R、G、B 通道的相似度
    sim_r = cosine_similarity(hist1_r, hist2_r)
    sim_g = cosine_similarity(hist1_g, hist2_g)
    sim_b = cosine_similarity(hist1_b, hist2_b)

    # 平均颜色相似度
    return (sim_r + sim_g + sim_b) / 3


def get_combined_similarity(image1, image2, model, processor, w1=0.5, w2=0.5):
    """
    计算两张图片的综合相似度，结合 CLIP 相似度和颜色相似度
    """
    # 获取 CLIP 相似度
    clip_similarity = get_clip_score(image1, image2, model, processor)
    
    # 获取颜色直方图和相似度
    hist1 = get_color_histogram_with_cuda(image1)
    hist2 = get_color_histogram_with_cuda(image2)
    color_similarity = calculate_color_similarity_with_cuda(hist1, hist2)
    
    # 计算最终相似度
    final_similarity = w1 * clip_similarity + w2 * (color_similarity * 100)  # 将颜色相似度映射到同一个尺度
    #print(final_similarity.cpu().item())
    #return final_similarity[0][0].astype(np.float64)
    return final_similarity.cpu().item()


if __name__=='__main__':
    
    dir= "sam2修正"
    for subfolder in os.listdir(dir):
        root_dir = os.path.join(dir, subfolder)        
        # 初始化 CLIP 模型和处理器
        clip_path = "clip-vit-large-patch14-336"
        model = CLIPModel.from_pretrained(clip_path)
        processor = CLIPProcessor.from_pretrained(clip_path)

        # 处理文件夹
        process_folder(root_dir, model, processor)
'''
    root_dir="Samseg"   
        # 初始化 CLIP 模型和处理器
    clip_path = "clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(clip_path)
    processor = CLIPProcessor.from_pretrained(clip_path)

        # 处理文件夹
    process_folder(root_dir, model, processor)
'''

