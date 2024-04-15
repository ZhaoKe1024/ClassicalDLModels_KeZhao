#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/6 19:39
# @Author: ZhaoKe
# @File : CatdogReader.py
# @Software: PyCharm

import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


class CatdogDataset(data.Dataset):

    def __init__(self, root, list_file, train, transform):
        print('data init')
        self.train = train
        self.root = root
        # self.image_size = 448
        self.image_size = 65
        self.mean = (123, 117, 104)  # RGB
        self.filenames = []
        self.labels = []
        self.transform = transform
        with open(list_file) as f:
            lines = f.readlines()
        # print(lines)
        for line in lines:
            # print(line)
            splited = line.strip().split(' ')
            self.filenames.append(splited[0])
            c = splited[1]
            self.labels.append(torch.tensor(int(c), dtype=torch.long))
        self.num_samples = len(self.labels)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # print(fname)
        # print("read image:", self.root + '/' +fname)
        img = cv2.imread(self.root + '/' + fname)
        # print(img)
        # print("img shape:", img.shape)
        labels = self.labels[idx].clone()
        # print(labels)
        if self.train:
            # img = self.random_bright(img)
            img = self.random_flip(img)
            img = self.randomScale(img)
            # img = self.randomBlur(img)
            # img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img = self.randomShift(img)
            img = self.randomCrop(img)
        # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()

        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug
        h, w, _ = img.shape
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)
        return img, labels

    def __len__(self):
        return self.num_samples

    def randomScale(self, bgr):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            return bgr
        return bgr

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            return im_lr
        return im

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomShift(self, bgr):
        # 平移变换

        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 > shift_y:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 <= shift_y:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]
            return after_shfit_image
        return bgr

    def randomCrop(self, bgr):
        if random.random() < 0.5:
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped
        return bgr

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    train_dataset = CatdogDataset(root="E:/DATAS/catdogclass/train", list_file='./catdog_train_list.txt', train=True,
                                transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)
    plt.figure(0)
    for i in range(100):
        img, target = next(train_iter)
        for j in range(16):
            plt.subplot(4,4,j+1)
            plt.imshow(img[j])
            plt.title(str(target[j]))
        break
    plt.show()
        # print(img.shape, target.shape)


if __name__ == '__main__':
    main()
