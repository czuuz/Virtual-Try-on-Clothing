# 虚拟试衣
#TODO：欢迎补充其他部分
#一页笔记太小，放不下太多内容，请尽量使用\[link text](https:// "title")的语法引入新建的CodiMD链接，或者：“\<details>
\<summary>显示的内容\</summary>要隐藏的内容” 的语法

## 项目
建议：请在每个模块下的README.md文件中说明该模块的使用方式，包括必要的环境配置、权重的下载、源代码的github仓库。提供注释

项目结构如下：  
├── README.md  
│  
├── CLIP  
│     
├── SAM     
│     
├── YOLO     
│         
├── mmpose #或者openpose  
│   
│  
└── video #动态配准的处理  
│     └── crawler.py #爬虫脚本   
├── yolo_done_by_syh
│ 
│
└── utils.py  

为了方便起见，我们统一把静态配准的数据集文件夹整理成如下形式，以夹克为例
老师给的原数据集:
```
夹克
├──4878342940501(一个图片组的标号)
│  ├──封面图
│  ├──衣服图
│  ├──主图
├──4938275336663
├──...
```
经过我们清洗与处理过的数据集:
```
夹克
├──4878342940501(一个图片组的标号)
│  ├──clothes
│  ├──person
├──4938275336663
├──...
```
## 进度安排

## 静态配准

### 用到的技术：
#### YOLO目标检测
#### SAM分割
#### CLIP
#### 人体姿态估计(Openpose)

## 动态配准

### 链接
[腾讯文档动态视频关键词链接](https://docs.qq.com/sheet/DQlpYSUhRWEV5Q3JF?tab=BB08J2)

[交大云盘链接](https://jbox.sjtu.edu.cn/l/w1UsNG)

[腾讯文档静态数据处理分工链接](https://docs.qq.com/sheet/DQlVQUG1obk9BRkV1)

[腾讯文档静态配准流程](https://docs.qq.com/doc/DQnd6S2RHTkNEWndF)
### 要求/标准
#TODO：对上传数据格式、关键词、去重、clip标准之类的要求和建议
1. 文件中有三条注释，分别是要求大家去调整自己想爬取的关键词，视频时长过滤，爬取页数

### 问题
#TODO：分享遇到的问题以及解决方案
1. 抖音网页版分成“综合”“视频”“用户”“直播”四栏，打开抖音网页版时默认是综合，此爬虫脚本需要在视频一栏下运行，所以替换的连接应该是先切换到视频一栏再搜索关键词的链接
2. 貌似两个字的关键词成功率很高，三个字的关键词有的可以跑出来有的跑不出来，会报错没有csv文件



#### CSV数据格式

<details>
<summary>例子：关于“美洋MEIYANG女装🍀”的视频。字段包括：</summary>

- 视频播放链接：https://www.douyin.com/aweme/v1/play/?video_id=v0300fg10000cse88v7og65semb4jlr0&line=0&file_id=5a45e8bc193843f697b7f3f23139a7e5&sign=fec0201c2a5626354fde00330306b7e4&is_play_url=1&source=PackSourceEnum_SEARCH
- 用户ID：1512552552475243
- 用户主页链接：https://www.douyin.com/user/MS4wLjABAAAAVwLKentexPYG7PUvx_WjNqy8jRwhoSBnlj4ommLql3xLeNBY-hM9jE7Fzf7-VcJN?relation=0&vid=7429962204629175602
- 视频ID：7429962204629175602
- 点赞数：281138
- 视频描述：福利款的外套姐妹一定不要错过，显瘦的外套而且穿起来很显白定制的颜色巨好看素颜穿都好看，看到姐妹捡漏！#回头率爆棚 #又美又飒 #谁穿谁好看不分年龄段 #美洋双11 #好会穿上新月
- 上传时间：2024-10-26 14:08:27
- 评论数：3
- 分享数：5
- 收藏数：7
- 是否为广告：0
- 是否为热门视频：1
- 视频播放链接（重复）：https://www.douyin.com/aweme/v1/play/?video_id=v0300fg10000cse88v7og65semb4jlr0&line=0&file_id=5a45e8bc193843f697b7f3f23139a7e5&sign=fec0201c2a5626354fde00330306b7e4&is_play_url=1&source=PackSourceEnum_SEARCH
</details>


