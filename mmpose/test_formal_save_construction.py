from detector import my_mmpose
import os 
import cv2
import numpy as np

image_folder_path = './washed_doupeng'  # 图片所在文件夹路径
output_root_path = './output'  # 输出根文件夹路径

detector = my_mmpose.MMPoseDetector()

# 阈值，可改
threshold_main = 11.9  # 大致分数阈值（不管全身半身）
threshold_half = 0.5  # 半身与全身
threshold_front = 3.9  # 正面朝向与否
threshold_ifface = 1  # 头有没有照进来

# 获取文件夹内所有图片文件名，支持多层文件夹
image_filenames = []
# sequence_numbers = set()  # 使用集合来储存序列号
for root, dirs, files in os.walk(image_folder_path):
    if 'person_wearing_clothes' in root:
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # 根据需要更改图片格式
                image_filenames.append(os.path.join(root, file))

# 开始处理每一张图片
for image_path in image_filenames:
    image = cv2.imread(image_path)  # 读取图片
    annoed, selected_instances = detector.get_landmarks(image, 'one')  # TIP
    # 走流程关掉 #
    #print(selected_instances) #
    #cv2.imshow('Annoed Image', annoed) #
    #cv2.waitKey(0) #
    #cv2.destroyAllWindows() #
    # 走流程关掉 #
    # 获取图片所属的序列号（父文件夹名称）
    sequence_number = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    
    '''# 设置输出路径
    output_entire_path = os.path.join(output_root_path, sequence_number, 'output_entire')
    output_half_path = os.path.join(output_root_path, sequence_number, 'output_half')
    output_noface_entire_path = os.path.join(output_root_path, sequence_number, 'output_noface_entire')
    output_noface_half_path = os.path.join(output_root_path, sequence_number, 'output_noface_half')

    # 创建输出目录
    os.makedirs(output_entire_path, exist_ok=True)
    os.makedirs(output_half_path, exist_ok=True)
    os.makedirs(output_noface_entire_path, exist_ok=True)
    os.makedirs(output_noface_half_path, exist_ok=True)
'''
    # 设置输出路径
    output_sequence_path = os.path.join(output_root_path, sequence_number)
    output_person_path = os.path.join(output_sequence_path, 'person')
    output_clothes_path = os.path.join(output_sequence_path, 'clothes')
    
    # 创建输出目录
    os.makedirs(output_sequence_path, exist_ok=True)
    os.makedirs(output_person_path, exist_ok=True)
    os.makedirs(output_clothes_path, exist_ok=True)  # 保留该文件夹

    # 在 person 中再创建输出四个分类文件夹
    output_entire_path = os.path.join(output_person_path, 'output_entire')
    output_half_path = os.path.join(output_person_path, 'output_half')
    output_noface_entire_path = os.path.join(output_person_path, 'output_noface_entire')
    output_noface_half_path = os.path.join(output_person_path, 'output_noface_half')

    os.makedirs(output_entire_path, exist_ok=True)
    os.makedirs(output_half_path, exist_ok=True)
    os.makedirs(output_noface_entire_path, exist_ok=True)
    os.makedirs(output_noface_half_path, exist_ok=True)

    if selected_instances and len(selected_instances) > 0 and hasattr(selected_instances[0], 'keypoint_scores'):
        keypoint_scores = selected_instances[0].keypoint_scores
        main_score = keypoint_scores[0, :-2].flatten()  # 大体得分
        front_score = keypoint_scores[0, :5].flatten()  # 前五个得分对应五官
        body_score = keypoint_scores[0, 5:-4].flatten()  # 除去头部五官和下身的得分
        
        # print(np.sum(front_score))
        # print(np.sum(main_score))
        if np.sum(main_score) > threshold_main:  # 大体可行，继续细判断
            if np.sum(front_score) > threshold_front:
                half_score = keypoint_scores[0, -2:].flatten()
                if all(score > threshold_half for score in half_score):
                    new_image_path = os.path.join(output_entire_path, os.path.basename(image_path))
                    cv2.imwrite(new_image_path, image)
                    print(f'{os.path.basename(image_path)} 此全身图已存入 {output_entire_path} 文件夹')
                else:  # 存半身图
                    new_image_path = os.path.join(output_half_path, os.path.basename(image_path))
                    cv2.imwrite(new_image_path, image)
                    print(f'{os.path.basename(image_path)} 此半身图已存入 {output_half_path} 文件夹')
            else:
                print(f'{os.path.basename(image_path)} 此图正面朝向不够')
        elif np.sum(front_score) < threshold_ifface:  # 没有人脸,但是可能衣服是正的
            # print(np.sum(body_score))
            if np.sum(body_score) > 7.5:  # 半身图
                half_score = keypoint_scores[0, -2:].flatten()
                if all(score > threshold_half for score in half_score):
                    new_image_path = os.path.join(output_noface_entire_path, os.path.basename(image_path))
                    cv2.imwrite(new_image_path, image)
                    print(f'{os.path.basename(image_path)} 此无人脸全身图已存入 {output_noface_entire_path} 文件夹')
                else:  # 存半身图
                    new_image_path = os.path.join(output_noface_half_path, os.path.basename(image_path))
                    cv2.imwrite(new_image_path, image)
                    print(f'{os.path.basename(image_path)} 此半身无人脸图已存入 {output_noface_half_path} 文件夹')
        else:
            print(f'{os.path.basename(image_path)} 此图正面朝向不够')
