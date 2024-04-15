#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/19 23:03
# @Author: ZhaoKe
# @File : cvae_conv.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdkit.models.autoencoder import ConvDecoder


class ConvVAE(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128,
                 latent_dim=8, conditional=False, num_labels=0):  # , class_num=23, class_num1=6):
        super(ConvVAE, self).__init__()
        self.encoder = ConvVAEEncoder(input_channel=input_channel,
                                      input_length=input_length,
                                      input_dim=input_dim,
                                      latent_dim=latent_dim,
                                      conditional=conditional, num_labels=num_labels)
        self.decoder = ConvVAEDecoder(input_channel=input_channel,
                                      input_length=input_length,
                                      input_dim=input_dim,
                                      latent_dim=latent_dim, max_cha=self.encoder.max_cha,
                                      conditional=conditional, num_labels=num_labels)

    def forward(self, input_mel, conds=None):
        _, means, logvar = self.encoder(input_mel, conds)
        # print("shape of latent map: ", latent_map.squeeze().shape)  # [64, 256, 33, 13]
        # print("shape of means logvar:", means.shape, logvar.shape)
        z = self.reparameterize(means, logvar)
        # print("shape of z: ", z.squeeze().shape)  # [64, 8, 33, 13]
        recon_mel = self.decoder(z, conds)
        # print("shape of recon:", recon_mel.shape)  # [64, 1, 288, 128]
        return recon_mel, z, means, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class ConvVAEEncoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128,
                 latent_dim=16, conditional=False, num_labels=0):  # , class_num=23, class_num1=6):
        super(ConvVAEEncoder, self).__init__()
        self.conditional = conditional
        self.num_labels = num_labels
        self.input_dim = input_channel
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        self.encoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1
        for i in range(len(es) - 2):
            self.encoder_layers.append(
                nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(
                nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        # self.z_len = input_length // 8 - 3
        # self.z_dim = input_dim // 8 - 3

        self.mean_linear = nn.Conv2d(self.max_cha + num_labels, latent_dim, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.var_linear = nn.Conv2d(self.max_cha + num_labels, latent_dim, kernel_size=1, stride=1, padding=0,
                                    bias=False)

    def forward(self, input_mel, conds):
        z = self.encoder_layers(input_mel)
        # print("z: ", z.squeeze().shape)
        if self.conditional:
            b, c, h, w = z.shape
            # print("zshape before concat:", z.shape)
            if conds.ndim == 1:
                conds_x = torch.zeros(size=(b, self.num_labels, h, w), device=input_mel.device)
                for i in range(len(z)):
                    conds_x[i, conds[i], :, :] = 1
            else:
                conds_x = torch.zeros(size=(b,))
                print("ndim=2?")
                # pass
            z = torch.concat((z, conds_x), dim=1)
            # print("zshape after concat:", z.shape)
        means = self.mean_linear(z)
        logvar = self.var_linear(z)
        return z, means, logvar


class ConvVAEDecoder(nn.Module):
    def __init__(self, max_cha, input_channel=1, input_length=288, input_dim=128, latent_dim=16
                 , conditional=False, num_labels=0):
        super(ConvVAEDecoder, self).__init__()
        self.decoder_projection = nn.Conv2d(latent_dim + num_labels, max_cha, kernel_size=1, stride=1, padding=0,
                                            bias=False)
        self.decoder_layers = ConvDecoder(input_channel=input_channel, input_length=input_length, input_dim=input_dim)
        self.conditional = conditional
        self.num_labels = num_labels

    def forward(self, latent_mel, conds):
        if self.conditional:
            b, c, h, w = latent_mel.shape
            # print("decoder zshape before concat:", latent_mel.shape)
            if conds.ndim == 1:
                conds_x = torch.zeros(size=(b, self.num_labels, h, w), device=latent_mel.device)
                for i in range(len(latent_mel)):
                    conds_x[i, conds[i], :, :] = 1
            else:
                conds_x = torch.zeros(size=(b,))
                print("ndim=2?")
                # pass
            latent_mel = torch.concat((latent_mel, conds_x), dim=1)
            # print("decoder zshape after concat:", latent_mel.shape)
        recon_input = self.decoder_projection(latent_mel)
        # print("proj after:", recon_input.shape)
        recon_mel = self.decoder_layers(recon_input)
        return recon_mel


def vae_loss_fn(recon_x, x, mean, log_var, kl_weight=0.00025):
    BCE = torch.nn.functional.mse_loss(
        recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # print(BCE.shape, KLD.shape)
    # kl_weight = 0.00025
    return (BCE + kl_weight * KLD) / x.size(0)


if __name__ == '__main__':
    # emodel = ConvVAEEncoder(conditional=True, num_labels=10)  # .to("cuda")
    # dmodel = ConvVAEDecoder(conditional=True, num_labels=10, max_cha=emodel.max_cha)
    model = ConvVAE(conditional=True, num_labels=23)
    # recon_mel = torch.rand(64, 1, 288, 128)
    input_mel = torch.rand(32, 1, 288, 128)  # .to("cuda")
    # input_mean = torch.rand(64, 128)
    # input_logvar = torch.rand(64, 128)
    # print(vae_loss_1(recon_mel, input_mel, input_mean, input_logvar))
    recon_x, z, me, lo = model(input_mel, conds=torch.randint(0, 23, size=(32,)))
    print(z.shape, me.shape, lo.shape)
    # recon_x = dmodel(z, conds=torch.randint(0, 10, size=(32,)))
    loss = vae_loss_fn(recon_x, input_mel, me, lo)
    print(loss)
    loss.backward()
    print(recon_x.shape)
