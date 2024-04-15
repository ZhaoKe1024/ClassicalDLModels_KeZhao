#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/14 13:25
# @Author: ZhaoKe
# @File : aeconv1_mnist.py
# @Software: PyCharm
import os
import random
import time
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dlkit.models.aeconv import Encoder, Decoder
from srutils import utils
from dlkit.modules.func import onehot


class TrainerCAEMNIST(object):
    def __init__(self, istrain=False):
        self.istrain = istrain
        self.configs = {
            "lr": 0.001, "weight_decay": 1e-5, "batch_size": 256, "d": 4, "fc_input_dim": 128, "seed": 3407,
            "epochs": 12, "iscond": True, "cond_dim": 10
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        # self.in_channels = 1
        # self.demo_test = demo_test
        self.run_save_dir = "../run/aeconv_mnist/" + self.timestr + 'conditional_mix/'
        if istrain:
            # if not self.demo_test:
            os.makedirs(self.run_save_dir, exist_ok=True)
        self.model_e, self.model_d = None, None
        self.optim, self.loss_fn = None, None
        self.train_loader, self.valid_loader = None, None
        self.test_dataset, self.test_loader = None, None
        self.mix_rate = int(0.667 * self.configs["batch_size"])

    def __setup_dataset(self):
        data_dir = 'F:/DATAS/mnist'
        if self.istrain:
            train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
            train_transform = transforms.Compose([transforms.ToTensor(), ])
            train_dataset.transform = train_transform
            if self.configs["iscond"]:
                train_dataset.target_transform = onehot(self.configs["cond_dim"])
            m = len(train_dataset)
            train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.configs["batch_size"])
            self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=self.configs["batch_size"])

        self.test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        test_transform = transforms.Compose([transforms.ToTensor(), ])
        self.test_dataset.transform = test_transform
        if self.configs["iscond"]:
            self.test_dataset.target_transform = onehot(self.configs["cond_dim"])
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.configs["batch_size"], shuffle=True)

        print("Built Dataset and DataLoader")

    def test_data(self):
        data_dir = 'F:/DATAS/mnist'
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        train_transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset.transform = train_transform
        train_dataset.target_transform = onehot(self.configs["cond_dim"])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.configs["batch_size"])

        for i, (x, y) in enumerate(self.train_loader):
            print(x.shape)
            print(y.shape)
            break
        # z = torch.randn(size=(128, 128))
        # y = torch.randint(0, 10, size=(128, 10))
        # out = torch.cat([z, y], dim=1)
        # print(out.shape)

    def data_mix(self, x, y):
        mix_x, mix_y = copy(x), copy(y)
        for j in range(x.shape[0]):
            if j < self.mix_rate:
                r = random.randint(1, 10)
                # print(r, y[j:j+r])
                mix_x[j] = torch.mean(x[j:j + r], dim=0)
                mix_y[j] = torch.sum(y[j:j + r], dim=0)
                # print(r, x[j].shape, x[j:j + r].shape, y[j])
        # print(x.shape)
        # print(y.shape)
        return mix_x, mix_y

    def setup_models(self):
        self.model_e = Encoder(encoded_space_dim=self.configs["d"], fc2_input_dim=self.configs["fc_input_dim"],
                               iscond=self.configs["iscond"], cond_dim=self.configs["cond_dim"])
        self.model_d = Decoder(encoded_space_dim=self.configs["d"], fc2_input_dim=self.configs["fc_input_dim"],
                               iscond=self.configs["iscond"], cond_dim=self.configs["cond_dim"])
        self.model_e.to(self.device)
        self.model_d.to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        if self.istrain:
            paras_to_optimize = [
                {"params": self.model_e.parameters()},
                {"params": self.model_d.parameters()}
            ]
            self.optim = torch.optim.Adam(paras_to_optimize, lr=self.configs["lr"], weight_decay=self.configs["weight_decay"])
        print("Built Model and Optimizer and Loss Function")

    def plot_ae_outputs(self, epoch_id):
        plt.figure()
        for i in range(5):
            ax = plt.subplot(2, 5, i+1)
            img = self.test_dataset[i][0].unsqueeze(0).to(self.device)
            self.model_e.eval()
            self.model_d.eval()
            with torch.no_grad():
                if self.configs["iscond"]:
                    y = self.test_dataset[i][1].unsqueeze(0).to(self.device)
                    rec_img = self.model_d(self.model_e(img, y), y)
                else:
                    rec_img = self.model_d(self.model_e(img))
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 5//2:
                ax.set_title("Original images")
            ax = plt.subplot(2, 5, i+1+5)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 5//2:
                ax.set_title("Reconstructed images")
        plt.savefig(self.run_save_dir+f"test_plot_epoch_{epoch_id}.png", format="png", dpi=300)

    def train_usl(self):
        """ 无监督，纯AutoEncoder"""
        utils.setup_seed(3407)
        self.__setup_dataset()
        self.setup_models()
        diz_loss = {"train_loss":[], "val_loss":[]}
        for epoch in range(self.configs["epochs"]):

            self.model_e.train()
            self.model_d.train()
            train_loss = []
            print(f"Train Epoch {epoch}")
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)

                if self.configs["iscond"]:
                    y = y.to(self.device)
                    mix_x, mix_y = self.data_mix(x, y)
                    encoded_data = self.model_e(mix_x, mix_y)
                    decoded_data = self.model_d(encoded_data, mix_y)
                    loss_value = self.loss_fn(mix_x, decoded_data)
                else:
                    encoded_data = self.model_e(x)
                    decoded_data = self.model_d(encoded_data)
                    loss_value = self.loss_fn(x, decoded_data)

                self.optim.zero_grad()
                loss_value.backward()
                self.optim.step()
                if i % 15 == 0:
                    print(f"\t partial train loss (single batch: {loss_value.data:.6f})")
                train_loss.append(loss_value.detach().cpu().numpy())
            train_loss_value = np.mean(train_loss)

            self.model_e.eval()
            self.model_d.eval()
            val_loss = 0.
            with torch.no_grad():
                conc_out = []
                conc_label = []
                for x, y in self.valid_loader:
                    x = x.to(self.device)
                    if self.configs["iscond"]:
                        y = y.to(self.device)
                        encoded_data = self.model_e(x, y)
                        decoded_data = self.model_d(encoded_data, y)
                    else:
                        encoded_data = self.model_e(x)
                        decoded_data = self.model_d(encoded_data)
                    conc_out.append(decoded_data.cpu())
                    conc_label.append(x.cpu())
                conc_out = torch.cat(conc_out)
                conc_label = torch.cat(conc_label)
                val_loss = self.loss_fn(conc_out, conc_label)
            val_loss_value = val_loss.data
            print(f"\t Epoch {epoch} test loss: {val_loss.item()}")
            diz_loss["train_loss"].append(train_loss_value)
            diz_loss["val_loss"].append(val_loss_value)
            torch.save(self.model_e.state_dict(), self.run_save_dir + '{}_epoch_{}.pth'.format("aeconve_cond", epoch))
            torch.save(self.model_d.state_dict(), self.run_save_dir + '{}_epoch_{}.pth'.format("aeconvd_cond", epoch))
            self.plot_ae_outputs(epoch_id=epoch)
        plt.figure(figsize=(10, 8))
        plt.semilogy(diz_loss["train_loss"], label="Train")
        plt.semilogy(diz_loss["val_loss"], label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.savefig(self.run_save_dir+"LossIter.png", format="png", dpi=300)

    def recon_test_one(self, resume_path):
        if (self.model_e is None) or (self.model_d is None):
            self.setup_models()
        self.model_e.eval()
        self.model_d.eval()
        state_dict_e = torch.load(os.path.join(resume_path, f'aeconve_cond_epoch_11.pth'))
        self.model_e.load_state_dict(state_dict_e)
        state_dict_d = torch.load(os.path.join(resume_path, f'aeconvd_cond_epoch_11.pth'))
        self.model_d.load_state_dict(state_dict_d)
        print(self.model_d)
        z = torch.randn(size=(10, 4, ), device=self.device)
        y_label = torch.zeros(size=(10, 10))
        labels = torch.arange(0, 10).unsqueeze(1)
        # print(labels)
        y_label.scatter_(1, labels, 1)
        y_label = y_label.to(self.device)
        recon_images = self.model_d(z, y_label)
        recon_images = recon_images.squeeze().detach().cpu().numpy()
        plt.figure(0)
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(recon_images[i])
            plt.xticks([])
            plt.yticks([])
            plt.title(f"generate_{i}")
        plt.savefig(resume_path+"generate_0.png", format="png", dpi=300)
        plt.show()


if __name__ == '__main__':
    trainer = TrainerCAEMNIST(istrain=True)
    # trainer.test_data()
    # trainer.train_usl()
    trainer.recon_test_one("../run/aeconv_mnist/202404141930conditional_mix/")

    # y = torch.randint(0, 10, size=(128,))
    # onehot = onehot(10)
    # out = onehot(y)
    #
    # print(out)
