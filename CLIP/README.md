本代码在SAM处理完成后的文件夹上继续进行处理，接受SAM处理后的图像文件夹，输出经过特征比对后的garment-person对和person-person对。garment-person对和person-person对以json脚本的方式存储在total_results_clothes和total_results_person中，每个类别下的每个组都包含一个json脚本。
接受输入格式为：
夹克
├──4878342940501(一个图片组的标号)
│  ├──clothes
│  ├──person
├──4938275336663
├──...

代码test_for_clothes_person2.py最终输出格式为：
夹克
├──4878342940501(一个图片组的标号)
│  ├──clothes_similarity_results.json
├──4938275336663
├──...

代码test_for_person_person2.py最终输出为：
夹克
├──4878342940501(一个图片组的标号)
│  ├──person_similarity_results.json
├──4938275336663
├──...


CLIP任务中共有两个代码文件：

test_for_clothes_person2.py用于形成garment-person对，结果的json文件保存在total_results_clothes文件夹中。

test_for_person_person2.py用于形成person-person对，结果的json文件保存在total_results_person文件夹中。

代码主要调用了clip模型的权重 以及增加了一个颜色对比模块 来最终决定两张图片的相似分数。
