#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/14 14:51
# @Author: ZhaoKe
# @File : vaeconv_cond.py
# @Software: PyCharm
from collections import OrderedDict

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


class VAE(nn.Module):
    def __init__(self, shape, nhid=16, iscond=False, cond_dim=10):
        super(VAE, self).__init__()
        self.dim = nhid
        self.iscond, self.cond_dim = iscond, cond_dim
        self.encoder = Encoder(shape, nhid, ncond=cond_dim)
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


class VAESVI(nn.Module):
    def __init__(self, shape, nhid=16, iscond=False, cond_dim=10):
        super(VAESVI, self).__init__()
        self.dim = nhid
        self.iscond, self.cond_dim = iscond, cond_dim
        self.encoder = Encoder(shape, nhid, ncond=cond_dim)
        self.decoder = Decoder(shape, nhid, ncond=cond_dim)

    def sampling(self, mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def model(self, x):
        """后验概率模型"""
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
                obs=x.reshape(-1, 784),
            )
            # return the loc so we can visualize it later
            return loc_img

    def guide(self, x):
        """简单分布，去逼近后验概率模型"""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            z = dist.Normal(z_loc, z_scale+1)
            pyro.sample("latent", z.to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img

        # define a helper function to sample from p(z) and output p(x|z)
    def sample_img(self, num_samples, return_z=False):
        # sample from p(z)
        z = self.z_prior(num_samples).sample()
        loc_img = self.decoder.forward(z)
        if return_z:
            return loc_img, z
        else:
            return loc_img

    def z_prior(self, num_samples):
        # sample from p(z)
        z_loc = torch.zeros(num_samples, self.z_dim)
        z_scale = torch.ones(num_samples, self.z_dim)
        z = dist.Normal(z_loc, z_scale)
        return z


class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid=16, ncond=16):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond=ncond)
        self.decoder = Decoder(shape, nhid, ncond=ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)

    def sampling(self, mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, class_idx, device=torch.device("cuda")):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)
        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res


BCE_loss = nn.BCELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight*KL_divergence


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
