# 动态视频生成

## 代码
https://github.com/novitalabs/AnimateAnyone
## 环境
```shell
conda create -n Animate python=3.10.16
pip install -r requirements.txt
pip install accelerate==0.21.0
```
## pose_videos
```shell
./anyone-video-2_kps.mp4
./anyone-video-5_kps.mp4
```
## 准备
获得预训练权重
```shell
python tools/download_weights.py
```
将data转换为png格式放到`./configs/inference/ref_images`，对应修改`./configs/prompts/animation.yaml`内容

## run
```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 64
```

## 结果
https://jbox.sjtu.edu.cn/l/j1iDSG
