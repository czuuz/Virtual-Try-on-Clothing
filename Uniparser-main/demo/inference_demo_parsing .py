from mmdet.apis import init_detector, inference_detector_parsing, show_result_pyplot
import mmcv
import numpy as np
import cv2
import os
from PIL import Image

from visual import *

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

home_root = "/root/multi-parsing/"
data_root = "../datasets/"
output_seg_file = "./output_seg/"
output_vis_file = "./output_vis/"

# config_file = './configs/repparsing/MHP_r50_fpn_half_gpu_1x_repparsing_DCN_fusion_metrics.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = './work_dirs/MHP_release_r50_fpn_8gpu_1x_repparsing_v0_DCN_fusion_metrics/epoch_12.pth'
config_file = '../configs/repparsing/CHIP_r101_fpn_half_gpu_6x_repparsing_DCN_fusion_metrics_noneg_cluster_light.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../ckpt/UniParser_CIHP_r101_light.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = data_root + '/TB2KHGUdwHqK1RjSZFgXXa7JXXa___1649925406.jpg'
img_list = os.listdir(data_root)
i = 0

for img in img_list:
    i += 1
    if i<9178:
        continue
    print(i)
    img_path = data_root + img
    
    result = inference_detector_parsing(model, img_path)
    if result == None:
        print(f"warning {img} doesn't have center")
        continue
    seg_masks = result[0]
    
    seg = seg_masks[0].astype(np.uint8)
    seg = Image.fromarray(seg)
    seg.save(output_seg_file + img)
    
    seg_vis = decode_labels(seg_masks)
    seg_vis = Image.fromarray(seg_vis[0])
    seg_vis.save(output_vis_file + img)
# gt_name = "7_02_02.png"

# result = inference_detector_parsing(model, img)

# seg_masks = result[0]
# if type(seg_masks) == list:
#     seg_masks = np.array(seg_masks)
# offset_vis = result[1]
# score_list = result[2]

# img_ori = mmcv.imread(img)
# h,w,_ = img_ori.shape

# seg = seg_masks[0].astype(np.uint8)



# image = Image.fromarray(seg) # 保存图片 
# image.save("output_image_seg.png")

# image_vis = decode_labels(seg_masks)
# image_vis = Image.fromarray(image_vis[0])
# image_vis.save("output_image_vis.png")

