#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/6 10:47
# @Author: ZhaoKe
# @File : eval.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from dlkit.models.capsnet import CapsNet

if __name__ == '__main__':
    # x = torch.rand(32, 128, 33, 31)
    # model = torch.nn.Linear(128, 32)
    # # new_x = F.pad(x, [2, 2, 2, 2, 0, 0], mode="constant", value=0)
    # print(model(x).shape)
    # print(new_x.shape)

    L = torch.eye(10).index_select(dim=0, index=torch.randint(0, 10, size=(16, )))
    print(L)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # caps = CapsNet().to(device)
    #
    # x = torch.rand(size=(16, 1, 28, 28)).to(device)
    # logi, recon = caps(x)
    # print(logi.shape, recon.shape)
    #
    # z = torch.randn(size=(16, 10, 16)).to(device)
    # y_label = torch.zeros(size=(16, 10))
    # labels = torch.randint(0, 10, size=(16, 1))
    # y_label.scatter_(1, labels, 1)
    # y_label = y_label.to(device)
    # print("one hot:")
    # print(y_label)
    #
    # recon = caps.generate(z=z, label=y_label)
    # print(recon.shape)
