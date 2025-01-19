所有文件名都遵从viton_hd的格式，因为文件太大，每个都分成了两部分压缩，称为first_half和second_half  
据我所知，训练只需要用到这些目前发的图片，我本地还保留一些中间的副产品，如有需要或者我遗漏了某些图片，我就再发过去  
- 衣服的图片不是原始图片，经过cloth_mask的处理
- agnostic系列的图片经过了一系列超参数的调整得到，如有什么问题可以修改
- 助教说在inpainting任务中，背景非常容易还原，所以没有给image文件夹做mask以去除背景
- person-cloth是我们这个作业额外加上的任务，它相当于cloth。为了与之前预训练时保持一致，助教建议从person身上抠出来cloth作为条件输入，而不是把整个人的图片输入进去。
- garment-person.txt相当于viton_hd中train.txt和test.txt的合体，没有区分训练还是测试，如果需要，可以先shuffle，再按百分比取出。每一行的左边是衣服，右边是人
- person_pair.json是person-person对的信息，之所以没有整理为garment-person.txt的格式，是因为person-person的排列组合太多，把数据集相应扩大了好几倍。训练的同学可以自行决定如何设置pair.txt的格式。
每一个列表的人物图片都是相似的  
此外，助教的原话是“鲤鱼:
person_person应该有两个视角的人

鲤鱼:
a视角的衣服让b视角的person穿

鲤鱼:
这样才能避免copy pasting现象”
我的理解是，在训练person-person对的时候，不要用同一个人的衣服作为输入条件。因为我们的衣服是直接从人身上扣下来的，如果输入的也是同一个人身上扣下来的衣服，那模型就直接把这部分贴上去就行了，走了捷径。

