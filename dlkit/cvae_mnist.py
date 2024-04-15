#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/16 15:44
# @Author: ZhaoKe
# @File : cvae_mnist.py
# @Software: PyCharm
import os
import time
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from dlkit.models.cvae_linear import LinearCVAE, vae_loss_fn
from dlkit.utils import utils


class TrainerCVAE(object):
    def __init__(self, istrain=True, conditional=False):
        self.configs = {
            "epochs": 10,
            "mnist_root": "F:/DATAS/mnist",
            "batch_size": 64,
            "learning_rate": 0.001,
            "encoder_layer_sizes": [784, 256],
            "decoder_layer_sizes": [256, 784],
            "latent_size": 2,
            "print_every": 100,
            "conditional": conditional
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        # self.in_channels = 1
        # self.demo_test = demo_test
        self.run_save_dir = "../run/cvae_mnist/" + self.timestr + '_conditional/'
        if istrain:
            # if not self.demo_test:
            os.makedirs(self.run_save_dir, exist_ok=True)

    def train(self):
        utils.setup_seed(3407)
        train_dataset = MNIST(root=self.configs["mnist_root"], train=True, transform=transforms.ToTensor(),
                              download=True)
        dataloader = DataLoader(train_dataset, batch_size=self.configs["batch_size"], shuffle=True)
        vae_model = LinearCVAE(encoder_layer_sizes=self.configs["encoder_layer_sizes"],
                               decoder_layer_sizes=self.configs["decoder_layer_sizes"],
                               latent_size=self.configs["latent_size"],
                               conditional=self.configs["conditional"],
                               num_labels=10 if self.configs["conditional"] else 0)
        vae_model.to(self.device)
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=self.configs["learning_rate"])
        logs = defaultdict(list)
        for epoch_id in range(self.configs["epochs"]):
            tracker_epoch = defaultdict(lambda: defaultdict(dict))
            for x_idx, (x_img, y_label) in enumerate(tqdm(dataloader, desc="Train|Epoch:")):
                x_img, y_label = x_img.to(self.device), y_label.to(self.device)
                if self.configs["conditional"]:
                    recon_x, z_mean, z_log_var, z_latent = vae_model(x_img, y_label)
                else:
                    recon_x, z_mean, z_log_var, z_latent = vae_model(x_img)

                for i, yi in enumerate(y_label):
                    x_id = len(tracker_epoch)
                    tracker_epoch[x_id]['x'] = z_latent[i, 0].item()
                    tracker_epoch[x_id]['y'] = z_latent[i, 1].item()
                    tracker_epoch[x_id]['label'] = yi.item()

                loss = vae_loss_fn(recon_x, x_img, z_mean, z_log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())

                if x_idx % self.configs["print_every"] == 0 or x_idx == len(dataloader) - 1:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch_id, self.configs["epochs"], x_idx, len(dataloader) - 1, loss.item()))
                    if self.configs["conditional"]:
                        c_code = torch.arange(0, 10).long().unsqueeze(1).to(self.device)
                        z_latent = torch.randn([c_code.size(0), self.configs["latent_size"]]).to(self.device)
                        recon_x = vae_model.inference(z_latent, c=c_code)
                    else:
                        z_latent = torch.randn([10, self.configs["latent_size"]]).to(self.device)
                        recon_x = vae_model.inference(z_latent)
                    plt.figure(x_idx)
                    plt.figure(figsize=(5, 10))
                    for p in range(10):
                        plt.subplot(5, 2, p + 1)
                        if self.configs["conditional"]:
                            plt.text(
                                0, 0, "c={:d}".format(c_code[p].item()), color='black',
                                backgroundcolor='white', fontsize=8)
                        plt.imshow(recon_x[p].view(28, 28).cpu().data.numpy())
                        plt.axis('off')
                    plt.savefig(self.run_save_dir + "E{:d}_{:d}.png".format(epoch_id, x_idx), dpi=300)
                    plt.clf()
                    plt.close('all')

                    # if not os.path.exists(self.run_save_dir+"")
            df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            g = sns.lmplot(
                x='x', y='y', hue='label', data=df.groupby('label').head(100),
                fit_reg=False, legend=True)
            g.savefig(self.run_save_dir + "E{:d}-Dist.png".format(epoch_id), dpi=300)


if __name__ == '__main__':
    cvae_trainer = TrainerCVAE(istrain=True, conditional=True)
    cvae_trainer.train()
