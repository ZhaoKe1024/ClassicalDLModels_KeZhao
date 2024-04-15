#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/19 11:01
# @Author: ZhaoKe
# @File : trianer_vae.py
# @Software: PyCharm
import os
import time
from datetime import timedelta
import json
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image

from dlkit.models.vae_vanilla import VanillaVAE


class VAETrainer(object):
    def __init__(self, config="../configs/aevae.yaml", istrain=False):
        with open(config, 'r', encoding='utf-8') as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_vae_vanilla/'
            if istrain:
                os.makedirs(self.run_save_dir, exist_ok=True)

    def __setup_dataloader(self, is_train):
        if is_train:
            self.train_dataset = MNIST("F:/DATAS/mnist", train=True, download=True, transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.5,), (0.5,))]))
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=64,
                                           shuffle=True, num_workers=0)
        # 获取测试数据
        self.valid_dataset = MNIST("F:/DATAS/mnist", train=False, download=True, transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.5,), (0.5,))]))
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=64,
                                       shuffle=True, num_workers=0)

    def __setup_model(self, is_train):
        self.model = VanillaVAE()
        self.model.to(self.device)
        # optimizer & scheduler
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=float(self.configs.optimizer_conf.learning_rate))

