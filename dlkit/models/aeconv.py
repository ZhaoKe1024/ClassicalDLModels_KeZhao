#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/14 13:23
# @Author: ZhaoKe
# @File : aeconv.py
# @Software: PyCharm
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim=128, iscond=False, cond_dim=10):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)  ### Linear section
        if iscond:
            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 32 + cond_dim, fc2_input_dim),
                nn.ReLU(True),
                nn.Linear(128, encoded_space_dim)
            )
        else:
            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 32, fc2_input_dim),
                nn.ReLU(True),
                nn.Linear(128, encoded_space_dim)
            )
        self.iscond=iscond

    def forward(self, x, cond_vec=None):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        if self.iscond:
            x = self.encoder_lin(torch.cat([x, cond_vec], dim=1))
        else:
            x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim=128, iscond=False, cond_dim=10):
        super().__init__()
        if iscond:
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim+cond_dim, fc2_input_dim),
                nn.ReLU(True),
                nn.Linear(128, 3 * 3 * 32),
                nn.ReLU(True)
            )
        else:
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, fc2_input_dim),
                nn.ReLU(True),
                nn.Linear(128, 3 * 3 * 32),
                nn.ReLU(True)
            )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),

            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )
        self.iscond = iscond

    def forward(self, x, cond_vec=None):
        if self.iscond:
            x = self.decoder_lin(torch.cat([x, cond_vec], dim=1))
        else:
            x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
