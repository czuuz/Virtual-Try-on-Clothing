import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt


def get_masked_cloth():
    cloth_folder = r"F:\datasets\virtual_tryon\ML_static\train\cloth"
    cloth_mask_folder = r"F:\datasets\virtual_tryon\ML_static\train\cloth_mask"
    new_cloth_folder = r"F:\datasets\virtual_tryon\ML_static\train\new_cloth"
    os.makedirs(new_cloth_folder, exist_ok=True)

    cloth_fn = os.listdir(cloth_folder)

    for fn in tqdm(cloth_fn):
        img = Image.open(os.path.join(cloth_folder, fn))
        mask = Image.open(os.path.join(cloth_mask_folder, fn)).convert(
            "L"
        )  # 确保掩码为灰度图
        # 创建一个与原始图像相同大小的白色背景图像
        white_bg = Image.new("RGB", img.size, "white")
        # 使用掩码来裁剪原始图像，得到只包含衣服的图像
        clothes_img = Image.composite(img, white_bg, mask)
        # 创建新的图像对象
        clothes_img.save(os.path.join(new_cloth_folder, fn))

def get_masked_person():
    person_folder = r"F:\datasets\virtual_tryon\ML_static\train\image"
    person_mask_folder = r"F:\datasets\virtual_tryon\ML_static\train\image-parse-v3_seg"
    new_person_folder = r"F:\datasets\virtual_tryon\ML_static\train\person-cloth"
    os.makedirs(new_person_folder, exist_ok=True)

    person_fn = os.listdir(person_folder)#[:50]

    for fn in tqdm(person_fn):
        # 加载原始图像
        img = cv2.imread(os.path.join(person_folder, fn))

        # 加载 DensePose 标签图 (假设为灰度图像)
        parse = cv2.imread(
            os.path.join(person_mask_folder, fn), cv2.IMREAD_GRAYSCALE
        )
        
        parse_array = np.array(parse)
        # print(np.unique(parse_array))
        parse_background = (parse_array == 0).astype(np.float32)
        parse_head = ((parse_array == 4).astype(np.float32) +
                        (parse_array == 13).astype(np.float32))#+
        parse_lower = (
            (parse_array == 9).astype(np.float32) +
            (parse_array == 12).astype(np.float32) +
            (parse_array == 16).astype(np.float32) +
            (parse_array == 17).astype(np.float32) +
            (parse_array == 18).astype(np.float32) +
            (parse_array == 19).astype(np.float32)
            )
        
        parse_body = (
            (parse_array == 1).astype(np.float32) +
            (parse_array == 2).astype(np.float32) +
            (parse_array == 3).astype(np.float32) +
            (parse_array == 11).astype(np.float32) +
            (parse_array == 14).astype(np.float32) +
            (parse_array == 15).astype(np.float32) 
        )

        parse_cloth = 1 - parse_background - parse_head - parse_lower - parse_body
        # 创建人体掩码
        human_mask = np.where(parse_cloth, 255, 0).astype(np.uint8)
        # plt.imshow(human_mask)
        # plt.show()
        # 创建白色背景图像
        white_bg = np.ones_like(img) * 255
        MORPH_SIZE = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_SIZE, MORPH_SIZE))
        # # 应用闭运算以填充孔隙
        human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_CLOSE, kernel)
        human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_OPEN, kernel)
        # plt.imshow(human_mask)
        # plt.show()
        # 将人体部分从原图中提取出来，并与白色背景合成
        human_only_img = np.where(human_mask[:, :, np.newaxis] == 255, img, white_bg) #img
        
        cv2.imwrite(os.path.join(new_person_folder, fn), human_only_img)
        

if __name__ == "__main__":
    get_masked_person()
"""     
在 CIHP (Chinese Internet Human Parsing) 数据集中，标签的定义用于对人体的不同部分进行分类。根据 CIHP 数据集的标准标签定义，标签值对应的人体不同部分如下：

0: Background (背景)
1: Hat (帽子)
2: Hair (头发)
3: Glove (手套)
4: Sunglasses (太阳镜)
5: UpperClothes (上衣)
6: Dress (连衣裙)
7: Coat (外套)
8: Socks (袜子)
9: Pants (裤子)
10: Jumpsuit (连体衣)
11: Scarf (围巾)
12: Skirt (裙子)
13: Face (脸部)
14: Left-arm (左臂)
15: Right-arm (右臂)
16: Left-leg (左腿)
17: Right-leg (右腿)
18: Left-shoe (左鞋)
19: Right-shoe (右鞋)
头发标签 """