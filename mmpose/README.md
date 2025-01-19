单独的那个test_formal_construction文件，它是正式使用的，用它替换掉原本文件夹中的test_pic.py

需要配置环境以及下载rtmo文件到ckpt目录下（由于上传文件大小限制无法上传）

set up the enviornment accordingly to this website https://mmpose.readthedocs.io/en/latest/installation.html

Acquire ckpt for mmpose via https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo and put it in `./ckpt`

代码使用只需要将test_formal_save_construction.py文件中的image_folder_path = './washed'  # 图片所在文件夹路径修改为想要清洗的图片文件夹即可

按照流程，接受的图片文件夹数据格式如下：
```
/夹克
|---488...（一组图片的标号）
|   |---clothes
|   |---person
|---4798...(另一组)
```

处理筛选得到正面朝向的穿衣模特图，最终保存的结果形式如下示例：
```
/output（eg:夹克）
|---488...（一组图片的标号）
|   |---clothes
|   |---person
|   |   |---output_entire
|   |   |---output_half
|   |   |---output_noface_half
|   |   |---output_noface_entire
|---4798...(另一组)
```

所分的四类为：output_entire————全身正面朝向图； output_half————上半身正面朝向图；  output_noface_half————无人脸上半身正面朝向图；  output_noface_entire————无人脸全身正面朝向图

另外那个process_clips_ver2.py是用于处理视频的openpose代码，作为动态配准部分平替原本test_formal_construction的视频帧处理代码。运行它即可处理你所写的视频文件夹里的所有视频，提取最佳帧
