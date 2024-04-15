#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/6 10:48
# @Author: ZhaoKe
# @File : create_data.py
# @Software: PyCharm
import os
import xml.etree.ElementTree as ET

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def create_konface_trainval_listtxt():
    anno_dir_path = "E:/DATAS/KonFace/train/Annotations/"
    name2idx = {"Yui": 0, "Mio": 1, "Mugi": 2, "Ritsu": 3, "Azusa": 4}
    with open("dlkit/data_utils/konface_train_val_list_single.txt", 'w') as tvf:
        for file_path_item in os.listdir(anno_dir_path):
            item_path = os.path.join(anno_dir_path, file_path_item).replace('\\', '/')
            file_obj = parse_rec(item_path)
            # line =
            for line_obj in file_obj:
                # print(line_obj)
                # print(" ".join([str(item) for item in line_obj['bbox']]))
                line = item_path.replace("Annotations","ImageSets").replace("xml", "jpg")+"#"+"#".join([str(item) for item in line_obj['bbox']])+"#"+str(name2idx[line_obj["name"]])
                # print(line)
                tvf.write(line+"\n")
    labels_dict = {
        0: "Yui", 1: "Mio", 2: "Mugi", 3: "Ritsu", 4: "Azusa"
    }
    with open(os.path.join("dlkit/data_utils/", 'konface_label_list.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(labels_dict)):
            f.write(f'{labels_dict[i]}\n')
def create_konface_test_listtxt1():
    anno_dir_path = "E:/DATAS/KonFace/test/Annotations/"
    name2idx = {"Yui": 0, "Mio": 1, "Mugi": 2, "Ritsu": 3, "Azusa": 4}
    with open("dlkit/data_utils/konface_test_list_single.txt", 'w') as tvf:
        for file_path_item in os.listdir(anno_dir_path):
            item_path = os.path.join(anno_dir_path, file_path_item).replace('\\', '/')
            file_obj = parse_rec(item_path)
            # line =
            for line_obj in file_obj:
                # print(line_obj)
                # print(" ".join([str(item) for item in line_obj['bbox']]))
                line = item_path.replace("Annotations","ImageSets").replace("xml", "jpg")+"#"+"#".join([str(item) for item in line_obj['bbox']])+"#"+str(name2idx[line_obj["name"]])
                # print(line)
                tvf.write(line+"\n")

def create_catdog_trainvalid_list():
    train_dir_path = "E:/DATAS/catdogclass/train"
    valid_dir_path = "E:/DATAS/catdogclass/val"
    test_dir_path = "E:/DATAS/catdogclass/test"

    with open("dlkit/data_utils/catdog_train_list.txt", 'w') as train_list_file:
        for file_path_item in os.listdir(train_dir_path+"/cat"):
            train_list_file.write("cat/"+file_path_item+" 0\n")
        for file_path_item in os.listdir(train_dir_path+"/dog"):
            train_list_file.write("dog/"+file_path_item + " 0\n")

    with open("dlkit/data_utils/catdog_valid_list.txt", 'w') as valid_list_file:
        for file_path_item in os.listdir(valid_dir_path+"/cat"):
            valid_list_file.write("cat/"+file_path_item+" 0\n")
        for file_path_item in os.listdir(valid_dir_path+"/dog"):
            valid_list_file.write("dog/"+file_path_item + " 0\n")

    with open("dlkit/data_utils/catdog_test_list.txt", 'w') as test_list_file:
        for file_path_item in os.listdir(test_dir_path+"/cat"):
            test_list_file.write("cat/"+file_path_item+" 0\n")
        for file_path_item in os.listdir(test_dir_path+"/dog"):
            test_list_file.write("dog/"+file_path_item + " 0\n")

if __name__ == '__main__':
    # create_konface_trainval_listtxt()
    # create_konface_test_listtxt1()
    create_catdog_trainvalid_list()
    # anno_path = "E:/DATAS/KonFace/train/Annotations/"
    # file_path = "1- (125).xml"
    # _file = os.path.join(anno_path, file_path).replace('\\', '/')
    # obj = parse_rec(_file)
    # for item in obj:
    #     print(item)