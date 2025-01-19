from detector import MMPoseDetector
import os
import cv2
import numpy as np

# video_folder_path = "D:/issue/动态clip/动态clip"  # 写你的视频文件夹路径
video_folder_path = './testvideo'
output_root_path = './output'  # 输出根文件夹路径

detector = MMPoseDetector()

if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

# 阈值，可改
threshold_main = 11.9  # 大致分数阈值（不管全身半身）
threshold_half = 0.5  # 半身与全身
threshold_front = 3.9  # 正面朝向与否
threshold_ifface = 1  # 头有没有照进来

# 获取文件夹内所有视频文件
video_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder_path, video_file)  # 生成完整视频路径
    print(f'处理视频: {video_path}')
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    # max_score = 11.4  # threshold_front = 3.9 + np.sum(body_score) > 7.5  
    max_score = 11.4
    best_frame = None


    while cap.isOpened():
        ret, frame = cap.read()
        # print(f'正在处理帧')
        if not ret:
            break

        annoed, selected_instances = detector.get_landmarks(frame, 'one')

        if selected_instances and len(selected_instances) > 0 and hasattr(selected_instances[0], 'keypoint_scores'):
            keypoint_scores = selected_instances[0].keypoint_scores
            # main_score = keypoint_scores[0, :-2].flatten()  # 大体得分 # 用不上
            front_score = keypoint_scores[0, :5].flatten()  # 前五个得分对应五官
            body_score = keypoint_scores[0, 5:-4].flatten() # 除去头部五官和下身的得分
            
            total_score = np.sum(body_score) + np.sum(front_score)
            if total_score > max_score:
                max_score = total_score
                best_frame = frame

    cap.release()

    if best_frame is not None:
        output_path = os.path.join(output_root_path, f'best_frame_{video_file}.jpg')
        cv2.imwrite(output_path, best_frame)
        print(f'最佳帧已保存到 {output_path}')
    else:
        print(f'未找到合适的帧在视频 {video_file}')
