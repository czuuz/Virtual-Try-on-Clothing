## 前言
下面用到的数据从这个交大云盘下载`https://jbox.sjtu.edu.cn/v/link/view/2d27f13e43f4491a9aa0c9ed11f63dcb`

以下的代码均在windows11上进行测试，如有问题欢迎随时联系我，不遵循下面的指示后果自负  
主要参考来自于`https://github.com/sangyun884/HR-VITON/issues/45`

- 整套流程开始前只有`train/image` `train/person`，请参考`MLdata/train`文件夹下的示例图片来确保你的图片处理后应该是什么样的
- 下载cloth.zip,image.zip，分别是衣服和人的图片
- 请切记命名规范，参考`MLdata/train`文件夹下，不要自作主张修改文件夹的名字，更不要修改图片的名字，否则就白做了
- 务必新开环境，否则后果自负
- 有任何问题都可以问我，最好在群里提出，因为大家可能都有这个问题

解压完文件后，应该如此安排文件位置：
```python
MLdata
|_ train_ image #人
       |_ cloth #衣服
```
github中包含了处理数据的示例，可以参考
### 数据处理步骤
共50000张人物图片，10000张衣服图片
分为
- openpose #3-4人 配置环境非常简单，我电脑上2秒一张图片
- human-parse #2人，可能在windows高版本的nvcc上跑不了，需要linux服务器环境
- dense-pose #2人
- cloth mask #1人，处理衣服的，上面三个是处理人的
上面的都是独立进行的，互不依赖。处理衣服可以不下载人，处理人可以不下载衣服
- parse agnostic #这部分不需要，删去
- human agnostic #依赖上面的前三项
- agnostic_mask #依赖上一项
这两个都不需要太多算力，我自己来试试


## openpose
使用windows portable version,必须下载交大云盘的openpose并解压，因为它的模型死活下载不了（如果你自己能搞定也可以）  
这里我不太敢担保我的openpose解压之后还能在你们的电脑上跑，可以参考https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md 自己安装也行
```bash
cd G:\storage\DL_lib\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose

# 不要修改下面的参数，除了路径
# 这里的openpose需要手部的关键点与json文件，如果你不确定你在做什么，请严格遵照下面的指示
bin\OpenPoseDemo.exe --image_dir F:\datasets\virtual_tryon\MLdata\train\image  --hand --disable_blending --display 0 --write_json F:\datasets\virtual_tryon\MLdata\train\openpose_json --write_images F:\datasets\virtual_tryon\MLdata\train\openpose_img --num_gpu 1 --num_gpu_start 0
```
想使用colab的可以参考上传的openpose.ipynb文件，但是不推荐，需要自己编译，速度极慢

## human parse
这部分我根本没法在windows上跑，可能因为nvcc版本太高了，建议使用linux服务器。  
下载云盘的CIHP_PGN-master和CIHP_pgn
参考https://github.com/Engineering-Course/CIHP_PGN

windows与linux服务器版本
```bash
conda create -n tf python=3.7
conda activate tf
conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6.5
pip install tensorflow-gpu==1.15
pip install scipy==1.7.3 opencv-python==4.5.5.62 protobuf==3.19.1 Pillow==9.0.1 matplotlib==3.5.1

cd storage/DL_lib/CIHP_PGN-master/CIHP_PGN-master #进入CIHP_PGN目录
# 建立从data/image路径到 CIHP_PGN-master/datasets/images路径的软链接
MKLINK /D "G:/storage/DL_lib/CIHP_PGN-master/CIHP_PGN-master/datasets/images" "F:/datasets/virtual_tryon/MLdata/train/image"
linux是： ln -s /lustre/home/acct-stu/stu234/machine_learning/MLdata/train/image /lustre/home/acct-stu/stu234/machine_learning/human_parse/CIHP_PGN-master/datasets/images #请修改为你的，使用相对路径可能会报错

# 建立模型权重的软链接，CIHP_pgn为权重
MKLINK /D "G:/storage/DL_lib/CIHP_PGN-master/CIHP_PGN-master/checkpoint" "G:/storage/DL_lib/CIHP_pgn/CIHP_pgn"
linux是： ln -s /lustre/home/acct-stu/stu234/machine_learning/human_parse/CIHP_pgn /lustre/home/acct-stu/stu234/machine_learning/human_parse/CIHP_PGN-master/checkpoint      #请修改为你的，使用相对路径可能会报错

python inf_pgn.py
# 我在windows跑不起来
```
kaggle版本

随便找一个有conda环境的notebook, https://www.kaggle.com/code/cjansen/conda
下载云盘的CIHP_PGN-master和CIHP_pgn，上传zip到kaggle，命名为humanparse之后点击create
```bash
!conda create -n tf python=3.7
!conda activate tf
!conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6.5
!pip install tensorflow-gpu==1.15
!pip install scipy==1.7.3 opencv-python==4.5.5.62 protobuf==3.19.1 Pillow==9.0.1 matplotlib==3.5.1

!ln -s input/humanparse/CIHP_PGN-master/CIHP_PGN-master/datasets/images  output/MLdata/train/image
!ln -s input/humanparse/CIHP_PGN-master/CIHP_PGN-master/checkpoint  input/humanparse/CIHP_pgn/CIHP_pgn

!cd input/humanparse/CIHP_PGN-master/CIHP_PGN-master
!python inf_pgn.py
```
这部分得到image-parse-v3
## dense pose
下载云盘上的`detectron2.zip` ，务必不要自己去找源代码下载，因为我的版本修改过源码，你也可以参照最上边的指引修改源码
```bash
conda create densepose -n python==3.8 #python>3.8的环境
conda activate densepose

#本人nvcc -V是12.3， 需要：
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

cd F:\github\CV #进入detectron2的根目录下
pip install -e detectron2  #使用开发者模型安装这个库

pip install av>=8.0.3 opencv-python-headless>=4.5.3.56 scipy>=1.5.4

#在detectron2\projects\DensePose目录下创建MLdata/image文件夹的软链接
MKLINK /D "F:\github\CV\detectron2\projects\DensePose\image_path" "F:\datasets\virtual_tryon\MLdata\train\image" 

#在detectron2\projects\DensePose下创建image-densepose文件夹,image-densepose里面再建一个image_path文件夹

cd detectron2/projects/DensePose
#F:\github\CV\detectron2\projects\DensePose目录下
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl image_path dp_segm -v
# 最后的文件会产生在`detectron2\projects\DensePose\image-densepose\image_path目录下，需要手动移动
```
这部分得到image-densepose文件夹
# cloth mask

从交大云盘上下载cloth_mask_model.zip并解压
```bash
#参考https://github.com/OPHoperHPO/image-background-remove-tool/tree/master
conda create -n cloth_mask python==3.9 #似乎要大于等于3.9
conda activate cloth_mask

# https://huggingface.co/Carve 模型地址
cd cloth_mask
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r cloth_mask_requirements.txt

# 模型下载不下来，下面是解决方案
# 务必把下面的代码修改成你的模型权重路径
H:\anaconda\envs\cloth_mask\Lib\site-packages\carvekit\ml\wrap\tracer_b7.py 51行
H:\anaconda\envs\cloth_mask\Lib\site-packages\carvekit\ml\wrap\fba_matting.py 六十多行

# 先创建MLdata/cloth-mask文件夹,请注意和代码所在的文件夹不一样
python cloth_mask\cloth_mask.py # 文件里的路径换成你的
```
这部分得到cloth-mask文件夹

# 最后处理
这部分基本是numpy的数组操作，不涉及神经网络，较快

## parse agnostic
修改data_path和output_path为你的路径
```bash
python parse_agnostic.py
```
得到image-parse-agnostic-v3.2，似乎这部分我们并不需要

## human agnostic
修改data_path和output_path为你的路径
```bash
python human_agnostic.py
```
得到agnostic-v3.2

## agnostic_mask
TODO  
得到agnostic-mask