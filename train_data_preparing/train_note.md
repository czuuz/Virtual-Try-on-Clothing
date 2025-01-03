## linux环境配置
直接按照README.md的指示应该就可以，需要先修改`environment.yml`的最后一行为你的路径
## windows环境配置
下载`xformers`时会出现文件名过长的问题，在`powershell`中执行以下命令解决：
```bash
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
尽量用conda安装
```bash
# 本人显卡3060, nvcc -V版本12.3
conda create -n ladi-vton -y python=3.10
conda activate ladi-vton
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install diffusers==0.14.0 transformers==4.27.3 accelerate==0.18.0 torchmetrics[image]==0.11.4 wandb==0.14.0 matplotlib==3.7.1 tqdm 
pip install opencv-python==4.7.0.72  clean-fid==0.1.35
pip install xformers==0.0.20 # 不安装这个版本会出问题
conda install huggingface_hub==0.12.0 # huggingface会报错
# https://pypi.org/project/huggingface-hub/#history 查阅兼容的版本
# https://pypi.org/project/diffusers/#history

# https://blog.csdn.net/weihuahello/article/details/139427431 按照triton来避免warning
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl
```

# 推理
```bash

cd github/Generation/ladi-vton
# export TORCH_HOME=/mnt/shared_disk/storage/torch_cache/ linux 设置torch缓存目录

python src/inference.py --dataset vitonhd --vitonhd_dataroot F:/datasets/virtual_tryon/zalando-hd-resized/ --output_dir ./output_dir --test_order paired --category upper_body --batch_size 2 --mixed_precision fp16 --enable_xformers_memory_efficient_attention --allow_tf32 --compute_metrics
# 需要开VPN 否则huggingface的模型下载不下来
```                       

--dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
--dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
--vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
--test_order <str>             test setting, options: ['paired', 'unpaired']
--category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
--output_dir <str>             output directory
--batch_size <int>             batch size (default=8)
--mixed_precision <str>        mixed precision (no, fp16, bf16) (default=no)
--enable_xformers_memory_efficient_attention <store_true>
                                enable memory efficient attention in xformers (default=False)
--allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
--num_workers <int>            number of workers (default=8)
--use_png <store_true>         use png instead of jpg (default=False)
--compute_metrics              compute metrics at the end of inference (default=False)

## 推理之后的评估
```bash
val_metrics ImportError: attempted relative import with no known parent package
解决方案：`from .generate_fid_stats import make_custom_stats`改为`from generate_fid_stats import make_custom_stats`

python src/utils/val_metrics.py --gen_folder output_dir --dataset vitonhd --vitonhd_dataroot F:/datasets/virtual_tryon/datasets-test/datasets/test --test_order paired --category lower_body 
```


## 训练
### 提取warp的衣服
```bash
# 注释掉 src\train_tps.py 中的
optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
start_epoch = state_dict['epoch']
把原来的
`if os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name, f"checkpoint_last.pth")):`
`state_dict = torch.load(os.path.join(args.checkpoints_dir, args.exp_name, f"checkpoint_last.pth"))`
中的"checkpoint_last.pth"换成你torch缓存路径的"warping_vitonhd.pth"
```
我们不训练这个模块，所以使用`--only_extraction`
```bash
python src/train_tps.py --dataset vitonhd  --vitonhd_dataroot F:\datasets\virtual_tryon\zalando-hd-resized --checkpoints_dir F:\storage\torch_cache\hub --exp_name checkpoints --batch_size 1  --only_extraction # 这部分自动对test的部分做处理？

    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     dataroot of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       dataroot of vitonhd dataset (required when dataset=vitonhd)
    --checkpoints_dir <str>        checkpoints directory
    --exp_name <str>               experiment name
    --batch_size <int>             batch size (default=16)
    --workers <int>                number of workers (default=10)
    --height <int>                 height of the input images (default=512)
    --width <int>                  width of the input images (default=384)
    --lr <float>                   learning rate (default=1e-4)
    --const_weight <float>         weight for the TPS constraint loss (default=0.01)
    --wandb_log <store_true>       log training on wandb (default=False)
    --wandb_project <str>          wandb project name (default=LaDI_VTON_tps)
    --dense <store_true>           use dense uv map instead of keypoints (default=False)
    --only_extraction <store_true> only extract the images using the trained networks without training (default=False)
    --vgg_weight <int>             weight for the VGG loss (refinement network) (default=0.25)
    --l1_weight <int>              weight for the L1 loss (refinement network) (default=1.0)
    --epochs_tps <int>             number of epochs for the TPS training (default=50)
    --epochs_refinement <int>      number of epochs for the refinement network training (default=50)
```

### 微调Unet
```bash
python src/train_vto.py --dataset vitonhd --vitonhd_dataroot F:\datasets\virtual_tryon\zalando-hd-resized --output_dir output_dir --inversion_adapter_dir <path>  --enable_xformers_memory_efficient_attention --use_clip_cloth_features --train_inversion_adapter --train_batch_size 2  --resume_from_checkpoint F:\storage\torch_cache\hub\checkpoints --cloth_input_type None --text_usage[这部分我们需要选择] --category[这部分我们需要选择]
```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory where the generated images will be written
    --save_name <str>              name of the generated images folder inside `output_dir`
    --test_order <str>             test setting, options: ['paired', 'unpaired']
    --unet_dir <str>               path to the UNet checkpoint directory. Should be the same as `output_dir` of the VTO training script
    --unet_name <str>              name of the UNet checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --inversion_adapter_dir <str>  path to the inversion adapter checkpoint directory. Should be the same as `output_dir` of the VTO training script. Needed only if `--text_usage` is set to `inversion_adapter`. (default=None)
    --inversion_adapter_name <str> name of the inversion adapter checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --emasc_dir <str>              path to the EMASC checkpoint directory. Should be the same as `output_dir` of the EMASC training script. Needed when --emasc_type!=none. (default=None)
    --emasc_name <str>             name of the EMASC checkpoint. To load the latest checkpoint, use `latest`. (default=latest)
    --pretrained_model_name_or_path <str>
                                   model identifier from huggingface.co/models (default=stabilityai/stable-diffusion-2-inpainting)
    --seed <int>                   seed for reproducible training (default=1234)
    --batch_size <int>             batch size(default=8)
    --allow_tf32 <store_true>      allow TF32 on Ampere GPUs (default=False)
    --enable_xformers_memory_efficient_attention <store_true>
                                   enable memory efficient attention in xformers (default=False)
    --num_workers <int>            number of workers (default=8)
    --category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
    --emasc_type <str>             type of EMASC, options: ['linear', 'nonlinear'] (default=nonlinear)
    --emasc_kernel <int>           kernel size for the EMASC module (default=3)
    --emasc_padding <int>          padding for the EMASC module (default=1)
    --text_usage <str>             text features to use, options: ['none', 'noun_chunks', 'inversion_adapter'] (default=inversion_adapter)
    --cloth_input_type <str>       cloth input type, options: ['none', 'warped'], (default=warped)
    --num_vstar <int>              number of predicted v* per image to use (default=16)
    --num_encoder_layers <int>     number of ViT layers to use in inversion adapter (default=1)
    --use_png <store_true>         use png instead of jpg (default=False)
    --num_inference_steps <int>    number of diffusion steps at inference time (default=50)
    --guidance_scale <float>       guidance scale of the diffusion (default=7.5)
    --use_clip_cloth_features <store_true>
                                   use precomputed clip cloth features instead of computing them each iteration (default=False).
    --compute_metrics              compute metrics at the end of inference (default=False)


## 代码术语
'image',cloth','pose_map', 'inpaint_mask', 'im_mask', 
'category'：'upper_body'  对于vton_hd只有这一类
'im_name'：人图片的名字, 'c_name': 衣服图片的名字，训练时二者合一，记作paired
模式：paired/unpair，指的是根据pair.txt文件，是否选取来自同一对的衣服和人，主要用于推理，unpaired更有挑战性，是没训练过的
'
inference 251行：agnostic = torch.cat([low_im_mask, low_pose_map], 1)

inference.py
252-266：功能是做warping，
279-286: text prompt需要改，原文只限制了三种类别
309行: cloth_input_type='warped',



train.py 192行
warped改为none，不做warping
parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
541行使用到了"noun_chunks"，表示caption
parser.add_argument("--text_usage", type=str, default='inversion_adapter',
                        choices=["none", "noun_chunks", "inversion_adapter"],
                        help="if 'none' do not use the text, if 'noun_chunks' use the coarse noun chunks, if "
                             "'inversion_adapter' use the features obtained trough the inversion adapter network")
从540行开始的部分需要修改关于prompt的部分



模仿vitonhd.py
vitonhd.py中VitonHDDataset含有：
caption_filename: str = 'vitonhd.json',我们是否需要对应的json