#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/16 12:02
# @Author: ZhaoKe
# @File : ccvaeflex.py
# @Software: PyCharm
from collections import OrderedDict

import torch
import torch.nn as nn
from modules.convtrans import conv_generator, convT_generator


class VAE(nn.Module):
    def __init__(self, shape, nhid=16, iscond=False, cond_dim=10):
        super(VAE, self).__init__()
        c, h, w = shape
        ww = ((w - 8) // 2 - 4) // 2
        hh = ((h - 8) // 2 - 4) // 2
        self.dim = nhid
        self.iscond, self.cond_dim = iscond, cond_dim
        self.encoder_conv = conv_generator(shape=(1, 288, 128))
        self.encoder_mlp = nn.Sequential(
            Flatten(), MLP([ww * hh * 64, 256, 128]))
        self.calc_mean = MLP([128 + cond_dim, 64, nhid], last_activation=False)
        self.calc_logvar = MLP([128 + cond_dim, 64, nhid], last_activation=False)
        self.decoder = Decoder(shape, nhid, ncond=cond_dim)

    def sampling(self, mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y=None):
        if self.iscond:
            mean, logvar = self.encoder(x, y)
            z = self.sampling(mean, logvar)
            recon = self.decoder(z, y)
        else:
            mean, logvar = self.encoder(x)
            z = self.sampling(mean, logvar)
            recon = self.decoder(z)
        return recon, mean, logvar

    def generate(self, batch_size=None, labels=None, device=torch.device("cuda")):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        if not self.iscond:
            res = self.decoder(z)
        else:
            res = self.decoder(z, labels)
        if not batch_size:
            res = res.squeeze(0)
        return res


BCE_loss = nn.BCELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight * KL_divergence


class Encoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w - 8) // 2 - 4) // 2
        hh = ((h - 8) // 2 - 4) // 2
        self.encode_conv = nn.Sequential(nn.Conv2d(c, 16, 5, padding=0), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                         nn.Conv2d(16, 32, 5, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, 2),
                                         nn.Conv2d(32, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, 2),
                                         )
        self.encode_mlp = nn.Sequential(
            Flatten(), MLP([ww * hh * 64, 256, 128]))
        self.calc_mean = MLP([128 + ncond, 64, nhid], last_activation=False)
        self.calc_logvar = MLP([128 + ncond, 64, nhid], last_activation=False)

    def forward(self, x, y=None):
        x = self.encode_conv(x)
        print("shape of feature map:", x.shape)
        x = self.encode_mlp(x)
        print("shape of latent vector:", x.shape)

        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            x = torch.cat((x, y), dim=1)
            # print(x.shape)
            return self.calc_mean(x), self.calc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid + ncond, 64, 128, 256, c * w * h], last_activation=False), nn.Sigmoid())

    def forward(self, z, y=None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class classifier(nn.Module):
    def __init__(self, input_dim, class_num: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.class_num = class_num
        self.mlp = MLP([input_dim])


if __name__ == '__main__':
    pass
