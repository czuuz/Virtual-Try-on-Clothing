""" 

目前所有54x69_100.jpg都被改名为420x420_90.jpg
"""

import os
import os.path as osp
import json
import shutil
from typing import List, Dict

import PIL.Image
import matplotlib.pyplot as plt

zh2en = {
    "polo衫": "polo_shirt",
    "T恤": "T-shirt",
    "内衣": "underwear",
    "卫衣": "sweatshirt",
    "吊带": "sling",
    "大衣": "coat",
    "夹克": "jacket",
    "套衫": "pullover",
    "打底": "base_layer",
    "披肩": "shawl",
    "抹胸": "tube_top",
    "斗篷": "cape",
    "无袖外套": "sleeveless_coat",
    "棉服": "cotton-padded_clothes",
    "泳装": "swimwear",
    "牛仔外套": "denim_jacket",
    "皮夹克": "leather_jacket",
    "皮毛大衣": "fur_coat",
    "皮毛短外套": "short_fur_coat",
    "皮革外套": "leather_coat",
    "礼服": "ceremonial_dress",
    "羽绒服": "down_jacket",
    "背心": "waistcoat",
    "衬衫": "shirt",
    "西服": "suit",
    "连体裤": "jumpsuit",
    "连衣裙": "dress",
    "针织外套": "knitted_coat",
    "风衣": "trench_coat",
    "马甲": "vest",
}

en2zh = {v: k for k, v in zh2en.items()}


def translate():
    """Translate the Chinese file names to English file names"""
    for file_name in os.listdir("washed"):
        try:
            os.rename(f"washed/{file_name}", f"washed/{zh2en[file_name]}")
        except:
            print(f"{file_name} not found")

    for file in os.listdir("json_file"):
        for file_name in os.listdir(f"json_file/{file}"):
            try:
                os.rename(
                    f"json_file/{file}/{file_name}",
                    f"json_file/{file}/{zh2en[file_name]}",
                )
            except:
                print(f"{file_name} not found")

    translation_dict = {}
    translation_dict["zh2en"] = zh2en
    translation_dict["en2zh"] = en2zh
    json.dump(
        translation_dict,
        open("translation_dict.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )


def plot_sim_pair(garment_path):
    """Plot the similar pair of text and image"""
    json_file = open(garment_path, "r", encoding="utf-8")
    garment_json = json.load(json_file)
    # person_json = open(person_path, "r", encoding="utf-8")
    root = os.path.dirname(os.path.dirname(garment_path))
    # pers = person_json.readlines()

    for im_name in garment_json:
        c_name = im_name

        im_path = os.path.join(root, "image", im_name)
        c_path = os.path.join(root, "cloth", c_name)
        image = PIL.Image.open(im_path)
        cloth = PIL.Image.open(c_path)
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(cloth)
        plt.show()


def sort_and_deduplicate_json(file_path):
    """这部分功能被整合到原来的部分中了，不需要单独调用"""
    with open(file_path, "r") as file:
        data = json.load(file)

    # 去重并排序
    original_len = len(data)
    unique_sorted_data = sorted(set(data))
    print(
        f"Original length: {original_len}, After deduplication: {len(unique_sorted_data)}"
    )

    with open(file_path, "w") as file:
        json.dump(unique_sorted_data, file, indent=4)


def merge_sort_gp_json(root):
    """
    garment-person: clothes_similarity_results.json
    if a garment/person img name occur in the json file, then add the img to the json file
    """
    garment_json_merged = []
    person_json_merged = []
    moded_root = os.path.join(root, "garment-person")

    for root_dir, dirs, files in os.walk(moded_root):

        for file in files:  # only one
            file_path = os.path.join(root_dir, file)
            with open(file_path, "r") as f:  # "clothes_similarity_results.json"
                similarity_results = json.load(f)
                for item in similarity_results:
                    images_name = item.get("clothes_image", [])
                    persons_name: List = item.get("person_image", [])

                    if not images_name:
                        continue

                    for image_name in images_name:
                        # print(image_name)
                        garment_json_merged.append(image_name)

                    for person_name in persons_name:
                        # print(person_name)
                        person_json_merged.append(person_name)
    garment_json_merged = sorted(list(set(garment_json_merged)))
    person_json_merged = sorted(list(set(person_json_merged)))
    print(f"====================merge_gp_json====================")
    print(f"add {len(garment_json_merged)} garment images")
    print(f"add {len(person_json_merged)} person images")
    with open(os.path.join(root, f"garment_merged.json"), "w") as f:
        json.dump(garment_json_merged, f, indent=4)

    with open(os.path.join(root, f"person_merged.json"), "w") as f:
        json.dump(person_json_merged, f, indent=4)


def merge_sort_pp_json(root):
    """
    WARNING: This function must be inplemented after merge_gp_json
    person-person: person_similarity_results.json
    if a person img name occur in the json file, then add the img to the json file
    """
    person_json_merged = []
    moded_root = os.path.join(root, "person-person")

    for root_dir, dirs, files in os.walk(moded_root):

        for file in files:  # only one
            file_path = os.path.join(root_dir, file)
            with open(file_path, "r") as f:
                similarity_results = json.load(f)
                for person_dict in similarity_results:
                    for person, images in person_dict.items():
                        person_json_merged.extend(images)
    person_json_merged = list(set(person_json_merged))
    print(f"====================merge_pp_json====================")
    print(f"add {len(person_json_merged)} person images")
    origin_path = os.path.join(root, f"person_merged.json")
    with open(origin_path, "r") as f:

        original_person_json = json.load(f)
        print(f"Before merge: {len(original_person_json)} person images")
        # merge two list
        original_person_json.extend(person_json_merged)
        original_person_json = sorted(list(set(original_person_json)))
    print(f"Total {len(original_person_json)} person images")
    with open(origin_path, "w") as f:
        json.dump(original_person_json, f, indent=4)


def remove_same_prefix():
    """
    最后所有文件的后缀都是_420x420_90.jpg
    对于jpg图片，删除具有同样前缀的文件，它们之间仅有尺寸不一样
    例如00a1a13a-4c58-4683-89ad-c679bf67c93f_54x69_100.jpg和00a1a13a-4c58-4683-89ad-c679bf67c93f_420x420_90.jpg
    同时更新json文件和图片文件夹
    """
    print("==============remove_same_prefix==============")
    with open("json_file\garment_merged.json", "r") as f:
        json_data = json.load(f)

        new_json_data = []
        for idx, line in enumerate(json_data):
            if line.endswith("_420x420_90.jpg"):
                new_json_data.append(line)

            elif line.endswith("_54x69_100.jpg"):
                line = line.replace("_54x69_100.jpg", "_420x420_90.jpg")
                new_json_data.append(line)
            else:  # other type
                # print(line)
                new_json_data.append(line)
        new_json_data = list(set(new_json_data))
        print(f"before: {len(json_data)}, after: {len(new_json_data)}")

    with open("json_file\garment_merged.json", "w") as f:
        json.dump(new_json_data, f, indent=4)

    with open("json_file\person_merged.json", "r") as f:
        json_data = json.load(f)

        new_json_data = []
        for idx, line in enumerate(json_data):
            if line.endswith("_420x420_90.jpg"):
                new_json_data.append(line)
                continue
            elif line.endswith("_54x69_100.jpg"):
                line = line.replace("_54x69_100.jpg", "_420x420_90.jpg")
                new_json_data.append(line)
            else:
                # print(line)
                new_json_data.append(line)
        new_json_data = list(set(new_json_data))
        print(f"before: {len(json_data)}, after: {len(new_json_data)}")

    with open("json_file\person_merged.json", "w") as f:
        json.dump(new_json_data, f, indent=4)


def copy_files(src_folder, dst_folder, mode="clothes_only", json_path=None):
    """
    遍历 src_folder 中的所有文件，如果在jsonfile中，将它们复制到 dst_folder 中。

    :param src_folder: 源文件夹路径（包含需要复制的文件）
    :param dst_folder: 目标文件夹路径（文件将被复制到这里）
    """
    # 确保目标文件夹存在，如果不存在则创建它
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 读取 json 文件，获取所有文件名
    json_file = open(json_path, "r").read()
    copy_cnt = 0
    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(src_folder):
        root_suffix = root.split("\\")[-1]  # 转义
        if root_suffix != mode:
            # print(f"File {root_suffix} is not {mode}, skipping")
            continue

        for file in files:
            # 构造完整文件路径

            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_folder, file)

            dst_file_path = dst_file_path.replace("_54x69_100.jpg", "_420x420_90.jpg")
            file = file.replace("_54x69_100.jpg", "_420x420_90.jpg")

            if file not in json_file:
                print(f"File {file} not in json file, skipping")
                continue

            if os.path.exists(dst_file_path):
                print(f"File {dst_file_path} already exists, skipping")
                continue
            # 复制文件到目标文件夹
            try:
                # os.remove(dst_file_path)
                shutil.copy2(src_file_path, dst_file_path)  # 使用 copy2 以保留元数据
                copy_cnt += 1
                print(f"Copied {src_file_path} to {dst_file_path}")
            except Exception as e:
                print(f"Failed to copy {src_file_path}: {e}")
    print(f"Total {copy_cnt} files copied")


def remove_unuseful_files(json_path, img_root):
    """如果文件不在对应的json文件中，则删除"""
    with open(json_path, "r") as f:
        json_data = json.load(f)
    remove_cnt = 0
    for img_name in os.listdir(img_root):
        if img_name not in json_data:
            remove_cnt += 1
            print(f"Removing {img_name}")
            os.remove(os.path.join(img_root, img_name))
    print(f"Total {remove_cnt} files removed")
    print(f"Left {len(os.listdir(img_root))} files")


def make_folder(root, mode="clothes_only"):
    cloth_types = os.listdir(root)
    for cloth_type in cloth_types:
        cloth_type_path = os.path.join(root, cloth_type)

        for file_name in os.listdir(cloth_type_path):  # num
            sim_pair_img = os.path.join(cloth_type_path, file_name, mode)

            if not os.path.exists(sim_pair_img):
                continue

            for img_name in os.listdir(sim_pair_img):
                img_path = os.path.join(sim_pair_img, img_name)
                img = PIL.Image.open(img_path)
                plt.imshow(img)
                plt.show()
                pass


def get_cloth_pair_txt(json_file_path, mode="garment-person"):
    cloth_json = open(osp.join(json_file_path, "garment_merged.json"), "r")
    cloth_data = json.load(cloth_json)
    
    json_file_path = os.path.join(json_file_path, mode)
    # print(json_file_path)
    pair_txts = []
    
    
    for root, dirs, files in os.walk(json_file_path):
        # print(files)
        for file in files:
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            # print(file_path)
            with open(file_path, "r") as f:
                json_data = json.load(f)  # list
                for item in json_data: #dict
                    for person in item["person_image"]: # list
                        cloth = item["clothes_image"][0].replace(
                            "_54x69_100.jpg", "_420x420_90.jpg"
                        ) #只有一个衣服
                        person = person.replace(
                            "_54x69_100.jpg", "_420x420_90.jpg"
                        )
                        if cloth not in cloth_data:
                            print(f"{cloth} not in cloth_data")
                            continue
                        
                        pair_txts.append(cloth +" " + person + "\n")
    print(f"Total {len(pair_txts)} pairs")
    pair_txts = sorted(list(set(pair_txts)))
    with open(f"{mode}.txt", "w") as f:
        f.writelines(pair_txts)
        
def get_person_pair_txt(json_file_path, mode="person-person"):
    image_name_lists = os.listdir(r"F:\datasets\virtual_tryon\ML_static\train\image")
    
    json_file_path = os.path.join(json_file_path, mode)
    # print(json_file_path)
    pair_datas: List[List] = []
    pair_data = []
    for root, dirs, files in os.walk(json_file_path):
        # print(files)
        for file in files:
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            # print(file_path)
            with open(file_path, "r") as f:
                json_data = json.load(f)  # list
                for item in json_data: #dict
                    pair_data = []
                    for img_names in item.values():
                        # print(img_name)
                        for img_name in img_names:
                            # print(img_name)
                            if img_name not in image_name_lists:
                                continue
                            
                            img_name = img_name.replace(
                                "_54x69_100.jpg", "_420x420_90.jpg"
                            )
                            pair_data.append(img_name)
                    pair_datas.append(pair_data)
    # print(f"Total {len(pair_txts)} pairs")
    # pair_txts = sorted(list(set(pair_txts)))
    with open(f"person_pair.json", "w") as f:
        json.dump(pair_datas, f, indent=4)

def visualize_pair(txt, img_root):
    with open(txt, "r") as f:
        lines = f.readlines()
    for line in lines:
        cloth, person = line.strip().split()
        cloth_path = os.path.join(img_root, "cloth", cloth)
        new_cloth_path = os.path.join(img_root, "new_cloth", cloth)
        person_path = os.path.join(img_root, "image", person)
        agnostic_mask_path = os.path.join(img_root, "agnostic-mask", person)
        agnostic_v32_path = os.path.join(img_root, "agnostic-v3.2", person)
        person_cloth_path = os.path.join(img_root, "person-cloth", person)
        
        cloth_img = PIL.Image.open(cloth_path)
        new_cloth = PIL.Image.open(new_cloth_path)
        person_img = PIL.Image.open(person_path)
        agnostic_mask = PIL.Image.open(agnostic_mask_path)
        agnostic_v32 = PIL.Image.open(agnostic_v32_path)
        person_cloth = PIL.Image.open(person_cloth_path)
        
        plt.subplot(231)
        plt.imshow(cloth_img)
        plt.title("cloth")
        plt.subplot(232)
        plt.imshow(person_img)
        plt.title("person")
        plt.subplot(233)
        plt.imshow(agnostic_mask)
        plt.title("agnostic-mask")
        plt.subplot(234)
        plt.imshow(agnostic_v32)
        plt.title("agnostic-v3.2")
        plt.subplot(235)
        plt.imshow(person_cloth)
        plt.title("person-cloth")
        plt.subplot(236)
        plt.imshow(new_cloth)
        plt.title("new_cloth")
        
        plt.show()
        
def select_some_pair(data_root, txt, dest_folder):
    with open(txt, "r") as f:
        lines = f.readlines()
    for line in lines:
        cloth, person = line.strip().split()
        cloth_path = os.path.join(data_root, "cloth", cloth)
        person_path = os.path.join(data_root, "image", person)
        cloth_img = PIL.Image.open(cloth_path)
        person_img = PIL.Image.open(person_path)
        plt.subplot(121)
        plt.imshow(cloth_img)
        plt.title(cloth+" "+person)
        plt.subplot(122)
        plt.imshow(person_img)
        plt.show()
        if input("y/n") == "y":
            shutil.copyfile(cloth_path, f"{dest_folder}/cloth/{cloth}")
            shutil.copyfile(person_path, f"{dest_folder}/image/{person}")
            print(f"copy {cloth} and {person} to selected folder")
        else:
            print("Skip")
    print("Done")

if __name__ == "__main__":
    # visualize_pair("garment-person.txt", r"F:\datasets\virtual_tryon\ML_static\train")
    
    # select_some_pair(r"F:\datasets\virtual_tryon\ML_static\train", "train_pairs.txt", r"F:\datasets\virtual_tryon\MLdata\train")
    vton_path = r"F:\datasets\virtual_tryon\zalando-hd-resized"
    mode = "test"

    # img_root = r"F:\datasets\virtual_tryon\ML_static\washed"
    # make_folder(img_root)

    modes = ["garment-person, person-person"]  # garment-person, person-person
    json_root = r"F:\datasets\virtual_tryon\ML_static\json_file"
    # get_person_pair_txt(json_root)
    
    # get_cloth_pair_txt(json_root, "garment-person")
    visualize_pair("garment-person.txt", r"F:\datasets\virtual_tryon\ML_static\train")
    # plot_sim_pair(osp.join(json_root, "garment_merged.json"))

    # handle json files
    # merge_sort_gp_json(json_root) #garment-person
    # merge_sort_pp_json(json_root) #person-person
    # for mode in modes:
    #     sort_and_deduplicate_json(osp.join(json_root,f'{mode.replace("-person", "")}_merged.json'))
    # remove_same_prefix()

    # based on json files to copy files
    src_folder = r"F:\datasets\virtual_tryon\ML_static\washed"
    dst_folder = r"F:\datasets\virtual_tryon\ML_static\train\image"
    # copy_files(src_folder, dst_folder, mode="person_wearing_clothes", json_path=osp.join(json_root, "person_merged.json"))# "clothes_only""person_wearing_clothes"
    # remove_unuseful_files(os.path.join(json_root, "person_merged.json"), dst_folder)
    pass
