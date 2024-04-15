#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/12 17:27
# @Author: ZhaoKe
# @File : image_folder.py
# @Software: PyCharm
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def group_path(dir):
    num_dirs = 0
    path = []
    for root, dirs, files in os.walk(dir):  # 遍历统计
        for name in dirs:
            num_dirs += 1
            path.append(os.path.join(root, name))
            path.sort()
    return num_dirs, path


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
