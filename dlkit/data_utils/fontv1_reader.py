#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/12 16:58
# @Author: ZhaoKe
# @File : fontv1_reader.py
# @Software: PyCharm
import os

import matplotlib.pyplot as plt
# import numpy as np

# import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import ImageFolder
# from torchvision import transforms
from torchvision import transforms as T
from image_folder import make_dataset, group_path
from PIL import Image
# from PIL import ImageFile
import random


# import fnmatch


class Fonts_imgfolder(Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''

    def __init__(self, root, transform=None, train=True):
        super(Fonts_imgfolder, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        # self.paths = make_dataset(self.root)
        if self.train:
            self.C_size = 52  # too much we fix it as the number of letters
            # letter 52
            self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))]
            # size 3
            self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
            self.Sizes = list(self.Sizes.keys())
            '''refer'''
            # color 10
            self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
                           'cyan': (0, 255, 255),
                           'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203),
                           'chocolate': (210, 105, 30),
                           'silver': (192, 192, 192)}
            self.Colors = list(self.Colors.keys())
            # style nearly over 100
            cates = None
            # print(os.path.join(self.root, 'a', 'medium', 'red', 'orange'))
            # print(os.walk(os.path.join(self.root, 'a', 'medium', 'red', 'orange')))
            for roots, dirs, files in os.walk(os.path.join(self.root, 'a', 'medium', 'red', 'orange')):
                cates = dirs
                break
            self.All_fonts = cates
            print(len(self.All_fonts))
            print(self.All_fonts, len(self.All_fonts))
        else:  # test mode
            self.C_size, self.paths = group_path(self.root)  # size of center image C

    def findN(self, index):
        # random choose a C image
        C_letter = self.Letters[index]
        C_size = random.choice(self.Sizes)
        C_font_color = random.choice(self.Colors)
        resume_colors = self.Colors.copy()
        resume_colors.remove(C_font_color)
        C_back_color = random.choice(resume_colors)
        C_font = random.choice(self.All_fonts)
        C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)
        ''' exclusive the C attribute avoid same with C'''
        temp_Letters = self.Letters.copy()  # avoid same size with C
        temp_Letters.remove(C_letter)
        temp_Size = self.Sizes.copy()  # avoid same size with C
        temp_Size.remove(C_size)
        temp_font_color = self.Colors.copy()  # avoid same font_color with C
        temp_font_color.remove(C_font_color)
        temp_back_colors = self.Colors.copy()  # avoid same back_color with C and avoid same color with font
        temp_back_colors.remove(C_back_color)
        temp_font = self.All_fonts.copy()  # avoid same font with C
        temp_font.remove(C_font)

        # A has same content
        '''SAME content'''
        A_letter = C_letter
        A_size = random.choice(temp_Size)
        A_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if A_font_color in resume_colors:
            resume_colors.remove(A_font_color)
        A_back_color = random.choice(resume_colors)
        A_font = random.choice(temp_font)
        A_img_name = A_letter + '_' + A_size + '_' + A_font_color + '_' + A_back_color + '_' + A_font + ".png"
        A_img_path = os.path.join(self.root, A_letter, A_size, A_font_color, A_back_color, A_font, A_img_name)

        # B has same size
        B_letter = random.choice(temp_Letters)
        '''SAME size'''
        B_size = C_size
        B_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if B_font_color in resume_colors:
            resume_colors.remove(B_font_color)
        B_back_color = random.choice(resume_colors)
        B_font = random.choice(temp_font)
        B_img_name = B_letter + '_' + B_size + '_' + B_font_color + '_' + B_back_color + '_' + B_font + ".png"
        B_img_path = os.path.join(self.root, B_letter, B_size, B_font_color, B_back_color, B_font, B_img_name)

        # D has same font_color
        D_letter = random.choice(temp_Letters)
        D_size = random.choice(temp_Size)
        '''SAME font_color'''
        D_font_color = C_font_color
        resume_colors = temp_back_colors.copy()
        if D_font_color in resume_colors:
            resume_colors.remove(D_font_color)
        D_back_color = random.choice(resume_colors)
        D_font = random.choice(temp_font)
        D_img_name = D_letter + '_' + D_size + '_' + D_font_color + '_' + D_back_color + '_' + D_font + ".png"
        D_img_path = os.path.join(self.root, D_letter, D_size, D_font_color, D_back_color, D_font, D_img_name)

        # E has same back_color
        E_letter = random.choice(temp_Letters)
        E_size = random.choice(temp_Size)
        resume_colors = temp_font_color.copy()
        resume_colors.remove(C_back_color)
        E_font_color = random.choice(resume_colors)
        '''SAME back_color'''
        E_back_color = C_back_color
        E_font = random.choice(temp_font)
        E_img_name = E_letter + '_' + E_size + '_' + E_font_color + '_' + E_back_color + '_' + E_font + ".png"
        E_img_path = os.path.join(self.root, E_letter, E_size, E_font_color, E_back_color, E_font, E_img_name)

        # F has same font
        F_letter = random.choice(temp_Letters)
        F_size = random.choice(temp_Size)
        F_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if F_font_color in resume_colors:
            resume_colors.remove(F_font_color)
        F_back_color = random.choice(resume_colors)
        '''SAME font'''
        F_font = C_font
        F_img_name = F_letter + '_' + F_size + '_' + F_font_color + '_' + F_back_color + '_' + F_font + ".png"
        F_img_path = os.path.join(self.root, F_letter, F_size, F_font_color, F_back_color, F_font, F_img_name)

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path

    def findtest(self, index):
        '''
                    refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style
                    A2B3D4E5F1_combine_2N
        A: size provider
        B: font_color provider
        D: back_color provider
        E: font provider
        F: letter provider
        '''
        group_path = self.paths[index]
        A_img_path = os.path.join(group_path, 'size.png')
        B_img_path = os.path.join(group_path, 'font_color.png')
        D_img_path = os.path.join(group_path, 'back_color.png')
        E_img_path = os.path.join(group_path, 'font.png')
        F_img_path = os.path.join(group_path, 'letter.png')

        return A_img_path, B_img_path, D_img_path, E_img_path, F_img_path

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        if self.train:
            A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path = self.findN(index)

            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            C_img = Image.open(C_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A_img = self.transform(A_img)
                B_img = self.transform(B_img)
                C_img = self.transform(C_img)
                D_img = self.transform(D_img)
                E_img = self.transform(E_img)
                F_img = self.transform(F_img)

            return {'A': A_img, 'B': B_img, 'C': C_img, 'D': D_img, 'E': E_img, 'F': F_img}
        else:  # test
            A_img_path, B_img_path, D_img_path, E_img_path, F_img_path = self.findtest(index)

            A_img = Image.open(A_img_path).convert('RGB')
            B_img = Image.open(B_img_path).convert('RGB')
            D_img = Image.open(D_img_path).convert('RGB')
            E_img = Image.open(E_img_path).convert('RGB')
            F_img = Image.open(F_img_path).convert('RGB')

            if self.transform is not None:
                A_img = self.transform(A_img)
                B_img = self.transform(B_img)
                D_img = self.transform(D_img)
                E_img = self.transform(E_img)
                F_img = self.transform(F_img)

            return {'A': A_img, 'B': B_img, 'D': D_img, 'E': E_img, 'F': F_img}

    def __len__(self):
        return self.C_size


def return_data(args):
    name = args.dataset
    batch_size = args.batch_size
    # crop_size = args.crop_size
    image_size = args.image_size
    train = args.train
    if train:
        num_workers = args.num_workers
    else:
        num_workers = 1  # test mode
    # Create dataset
    if name.lower() == 'fonts':
        if train:  # train mode
            root = args.dataset_path
        else:  # test mode
            root = os.path.join(args.test_img_path, name.lower())
        if not os.path.exists(root):
            print('No fonts dataset')
        transform = []
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = Fonts_imgfolder(root, transform, train)
    else:
        raise NotImplementedError

    # Create dataloader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=train,
                             num_workers=num_workers)

    return data_loader


if __name__ == '__main__':
    import numpy as np

    font_dataset = Fonts_imgfolder(root="G:/DATAS/fonts-v1/fonts-v1 (2)/fonts-v1")
    print(len(font_dataset))
    imgs = font_dataset[2]
    print(imgs)
    plt.figure()
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.imshow(imgs[chr(64+i)])
    plt.show()
    # data_loader = return_data()
