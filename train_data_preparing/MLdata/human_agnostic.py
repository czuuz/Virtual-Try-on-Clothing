import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))#+
    parse_lower = (
        (parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    white_image = Image.new('RGB', img.size, 'black')
    agnostic_mask = white_image.copy()
    agnostic_draw = ImageDraw.Draw(agnostic_mask)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'white', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'white', 'white')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'white', 'white')
    
    # 处理掩码
    parse_head = np.array(parse_head)
    # 定义结构元素（例如，使用椭圆形结构元素）
    MORPH_SIZE = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    # 应用闭运算以填充孔隙
    parse_head = cv2.morphologyEx(parse_head, cv2.MORPH_CLOSE, kernel)
    parse_head = cv2.morphologyEx(parse_head, cv2.MORPH_OPEN, kernel)
    
    agnostic_mask.paste(white_image, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic_mask.paste(white_image, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    
    # 处理掩码
    agnostic_mask = np.array(agnostic_mask)
    # 定义结构元素（例如，使用椭圆形结构元素）
    MORPH_SIZE = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
    # 应用闭运算以填充孔隙
    agnostic_mask = cv2.morphologyEx(agnostic_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    agnostic_mask = cv2.morphologyEx(agnostic_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # 将闭运算后的掩码转换回 PIL 图像
    agnostic_mask = Image.fromarray(agnostic_mask)
    
    # agnostic_mask.paste(white_image, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    # 原图施加mask
    agnostic = np.where(np.array(agnostic_mask) == 255, 128, np.array(img))
    agnostic = Image.fromarray(np.uint8(agnostic))
    
    person_cloth_mask = np.where(np.array(agnostic_mask) == 255, np.array(img), 255)
    person_cloth_mask = Image.fromarray(np.uint8(person_cloth_mask))

    return agnostic, agnostic_mask, person_cloth_mask

if __name__ =="__main__":
    data_path = r"F:\datasets\virtual_tryon\ML_static\train"
    output_path = r"F:\datasets\virtual_tryon\ML_static\train\agnostic-v3.2"
    mask_output_path = r"F:\datasets\virtual_tryon\ML_static\train\agnostic-mask"
    person_cloth_mask_output_path = r"F:\datasets\virtual_tryon\ML_static\train\person_cloth"
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    os.makedirs(person_cloth_mask_output_path, exist_ok=True)
    images = os.listdir(osp.join(data_path, 'image'))
    for im_name in tqdm(images):
        
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name#.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3_seg', label_name))

        agnostic, agnostic_mask, person_cloth_mask = get_img_agnostic(im, im_label, pose_data)
        
        agnostic.save(osp.join(output_path, im_name))
        agnostic_mask.save(osp.join(mask_output_path, im_name))
        person_cloth_mask.save(osp.join(person_cloth_mask_output_path, im_name))
        
