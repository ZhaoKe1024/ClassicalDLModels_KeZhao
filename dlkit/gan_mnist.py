#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/3 13:08
# @Author: ZhaoKe
# @File : gan_mnist.py
# @Software: PyCharm
import os
import matplotlib.pyplot as plt
import yaml
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image

from dlkit.models.simplegd import generator, discriminator
from dlkit.modules.func import setup_seed


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


class TrainLinGAN(object):
    def __init__(self, configs="../configs/gan.yaml", istrain=True, demo_test=False):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # load base_directory list
        setup_seed(3407)
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = demo_test
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_lingan_mnist/'
            if not self.demo_test:
                os.makedirs(self.run_save_dir, exist_ok=True)

    def train(self):
        G_model = generator().to(self.device)
        D_model = discriminator().to(self.device)
        criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.0003)
        g_optimizer = torch.optim.Adam(G_model.parameters(), lr=0.0003)

        train_dataset = MNIST("F:/DATAS/mnist", train=True, download=True, transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # 这四个存储所有epoch内的loss
        history1 = []
        # history2 = []
        z_dimension = 50
        D_model.train()
        G_model.train()
        num_epoch = 1000
        g_loss_list = []
        d_loss_list = []
        for epoch in range(num_epoch):
            for x_idx, (x_img, _) in enumerate(tqdm(dataloader, desc="Training")):
                num_img = x_img.size(0)
                x_img = x_img.view(num_img, -1)

                real_img = Variable(x_img).to(self.device)
                real_label = Variable(torch.ones(num_img, 1)).to(self.device)
                fake_label = Variable(torch.zeros(num_img, 1)).to(self.device)
                real_out = D_model(real_img)
                # print("shape of out:", real_out.shape, real_label.shape)
                d_loss_real = criterion(real_out, real_label)

                real_scores = real_out

                z = Variable(torch.randn(num_img, z_dimension)).to(self.device)
                fake_img = G_model(z)
                fake_out = D_model(fake_img)
                d_loss_fake = criterion(fake_out, fake_label)

                fake_scores = fake_out

                d_loss = d_loss_real + d_loss_fake
                d_loss_list.append(d_loss.item())
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                z = Variable(torch.randn(num_img, z_dimension)).to(self.device)
                fake_img = G_model(z)
                output = D_model(fake_img)
                g_loss = criterion(output, real_label)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_loss_list.append(g_loss.item())
                if (x_idx + 1) % 50 == 0:
                    print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                          'D real: {:.6f},D fake: {:.6f}'.format(
                        epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                        real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
                    ))
                if epoch == 0 and x_idx == len(dataloader) - 1:
                    real_images = to_img(real_img.cuda().data)
                    save_image(real_images, self.run_save_dir + 'real_images.png')
                if x_idx == len(dataloader) - 1:
                    fake_images = to_img(fake_img.cuda().data)
                    save_image(fake_images, self.run_save_dir + 'fake_images-{}.png'.format(epoch + 1))
                    plt.figure(x_idx)
                    plt.plot(range(len(g_loss_list)), g_loss_list, c="red")
                    plt.plot(range(len(d_loss_list)), d_loss_list, c="blue")
                    plt.xlabel("iteration")
                    plt.ylabel("losses")
                    plt.savefig(self.run_save_dir + f'loss_iter_{epoch}.png')
                    plt.close(x_idx)
        torch.save(G_model.state_dict(), self.run_save_dir + 'generator.pth')
        torch.save(D_model.state_dict(), self.run_save_dir + 'discriminator.pth')


if __name__ == '__main__':
    lgan = TrainLinGAN(istrain=True, demo_test=False)
    lgan.train()
