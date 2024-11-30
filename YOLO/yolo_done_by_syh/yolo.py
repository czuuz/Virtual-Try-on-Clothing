from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw
import os
import torch
from tqdm import tqdm
import pickle  # 导入pickle模块


# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载预训练的 YOLOv8 模型
model = YOLO('yolo11m.pt').to(device)  
model_2 = YOLO('deepfashion2_yolov8s-seg.pt').to(device)

# 类别标签定义
# 假设类别 0 是平铺服装，类别 1 是人身服装
CLASS_NAMES = model.names
CLASS_NAMES_2 = model_2.names

# 设置输入图片路径和输出路径
input_folder = Path("D:\suyihan\MachineLearning\dzy\TB\TB")  
out_folder = Path("D:\suyihan\MachineLearning\dzy\washed")
out_folder.mkdir(exist_ok=True)

folders = list(input_folder.glob('*'))
params_dict = {}

def detect_cloth(results, CLASS_NAMES, img, folder_name, image_path, output_folder, params_dict):
        # draw = ImageDraw.Draw(img)

        width, height = img.size  # 获取图片尺寸
        
        clothes_boxes = []
        total_clothes_area = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]
            # 根据实际类别名称调整条件
            if 'clothes' in label.lower() or 'dress' in label.lower() or 'shirt' in label.lower():
                xyxy = box.xyxy[0].tolist()  # 获取边界框坐标 [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                total_clothes_area += area
                clothes_boxes.append({'bbox': xyxy, 'area': area})

        # 如果没有检测到衣服，跳过
        if len(clothes_boxes) != 1:
            return

        # 检查衣服主体是否占据图片大部分（例如超过50%）
        image_area = width * height
        if (total_clothes_area / image_area) < 0.4:
            return

        # 检查衣服主体是否不重叠
        def boxes_overlap(box1, box2):
            x11, y11, x12, y12 = box1
            x21, y21, x22, y22 = box2
            if x11 >= x22 or x21 >= x12:
                return False
            if y11 >= y22 or y21 >= y12:
                return False
            return True

        overlap = False
        for i in range(len(clothes_boxes)):
            for j in range(i + 1, len(clothes_boxes)):
                if boxes_overlap(clothes_boxes[i]['bbox'], clothes_boxes[j]['bbox']):
                    overlap = True
                    break
            if overlap:
                break

        if overlap:
            return

        # # 保存原始图片到对应文件夹
        # try:
        #     img.save(output_folder / image_path.name)
        # except:
        #     return


        # 提取边界框参数并保存
        for idx, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]
            xyxy = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]

            # 计算中心点坐标和宽高
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            dict_key = f"{folder_name}_{image_path.name}"

            # # 绘制边界框
            # draw.rectangle(xyxy, outline='blue', width=2)
            # draw.text((xyxy[0], xyxy[1] - 10), label, fill='blue')

            # 将参数存入字典
            params_dict[dict_key] = {
                'Label': label,
                'Center X': center_x,
                'Center Y': center_y,
                'Width': width,
                'Height': height
            }
        
        # 保存原始图片到对应文件夹
        try:
            img.save(output_folder / image_path.name)
        except:
            return

def detect_person(results, CLASS_NAMES, results_2, CLASS_NAMES_2, img, folder_name, image_path, output_folder, params_dict):
        # # 创建绘图对象
        # draw = ImageDraw.Draw(img)

        width, height = img.size  # 获取图片尺寸
        person_boxes = []
        total_person_area = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]
            if label == 'person':  # 根据模型的实际类别名称调整
                xyxy = box.xyxy[0].tolist()  # 获取边界框坐标 [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                total_person_area += area
                person_boxes.append({'bbox': xyxy, 'area': area})
        # 检查是否只有一个人
        if len(person_boxes) != 1:
            return

        # 检查人主体是否占据图片大部分（例如超过50%）
        image_area = width * height
        if (total_person_area / image_area) < 0.4:
            return

        ## 检查人的边界框是否不重叠
        def boxes_overlap(box1, box2):
            x11, y11, x12, y12 = box1
            x21, y21, x22, y22 = box2
            if x11 >= x22 or x21 >= x12:
                return False
            if y11 >= y22 or y21 >= y12:
                return False
            return True

        overlap = False
        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                if boxes_overlap(person_boxes[i]['bbox'], person_boxes[j]['bbox']):
                    overlap = True
                    break
            if overlap:
                break

        if overlap:
            return
        #####################################

        # try:
        #     img.save(output_folder / image_path.name)
        # except:
        #     return

        # 提取边界框参数并保存
        for idx, box in enumerate(results_2[0].boxes):
            cls_id = int(box.cls[0])
            if cls_id == 8 or cls_id == 7:
                continue
            label = CLASS_NAMES_2[cls_id]
            xyxy = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]

            # 计算中心点坐标和宽高
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            dict_key = f"{folder_name}_{image_path.name}"

            # # 绘制边界框
            # draw.rectangle(xyxy, outline='blue', width=2)
            # draw.text((xyxy[0], xyxy[1] - 10), label, fill='blue')

            params_dict[dict_key] = {
                'Label': label,
                'Center X': center_x,
                'Center Y': center_y,
                'Width': width,
                'Height': height
            }
        # 保存原始图片到对应文件夹
        try:
            img.save(output_folder / image_path.name)
        except:
            return


for input_folder in tqdm(folders, desc="处理文件夹"): #input_folder :衣服名
    cloth_name = input_folder.name
    subfolders = list(os.listdir(input_folder))
    params_dict = {}

    for folder_name in tqdm(subfolders, desc=f"处理 {cloth_name} 中的子文件夹", leave=False): #folder_name :图片组名
        folder_path = input_folder / folder_name

        if len(os.listdir(folder_path)) == 0:
            continue
        output_folder = out_folder / cloth_name
        output_folder.mkdir(exist_ok=True)
        output_folder = output_folder / folder_name
        if os.path.exists(output_folder):
            continue
        output_folder.mkdir(exist_ok=True)


        # 获取所有图片文件以显示进度
        images = list(folder_path.glob('**/*.jpg'))

        for image_path in images:  # 假设图片格式为 jpg, image_path :图片名
            # 加载图片
            try:
                img = Image.open(image_path)
            except:
                continue
            # img = Image.open(image_path)
            if img.mode == 'RGBA':
                continue

            # 运行模型进行检测
            results = model(img, conf=0.85, verbose=False)
            results_2 = model_2(img, verbose=False)

            # 解析结果，检查是否包含“人”
            contains_person = any(int(box.cls[0]) == 0 for box in results[0].boxes)

            # 根据是否包含“人”来分类
            if contains_person:
                category = 'person_wearing_clothes'  # 有人穿着衣物
                category_folder = output_folder / category
                category_folder.mkdir(exist_ok=True)
                detect_person(results, CLASS_NAMES, results_2, CLASS_NAMES_2, img, folder_name, image_path, category_folder, params_dict)
            else:
                category = 'clothes_only'  # 只有衣物
                category_folder = output_folder / category
                category_folder.mkdir(exist_ok=True)
                detect_cloth(results_2, CLASS_NAMES_2, img, folder_name, image_path, category_folder, params_dict)
            # img.save(category_folder / image_path.name)
    params_file = out_folder/ cloth_name / 'params.pkl'
    with open(params_file, 'wb') as f:
        pickle.dump(params_dict, f)

print("分类完成。所有图片已分类保存。")



