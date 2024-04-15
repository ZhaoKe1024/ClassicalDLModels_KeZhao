#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/8 16:53
# @Author: ZhaoKe
# @File : capsnet_mnist.py
# @Software: PyCharm
import os
import time
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from collections import defaultdict
from dlkit.models.capsnet import CapsNet, CapsuleLoss
from dlkit.utils import utils


class TrainerCapsNet(object):
    def __init__(self, istrain=True):
        self.configs = {
            "epochs": 10,
            "mnist_root": "F:/DATAS/mnist",
            "batch_size": 128,
            "learning_rate": 0.001,
            "print_every": 150,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        # self.in_channels = 1
        # self.demo_test = demo_test
        self.run_save_dir = "../run/capsnet_mnist/" + self.timestr+"/"
        if istrain:
            # if not self.demo_test:
            os.makedirs(self.run_save_dir, exist_ok=True)
            with open(self.run_save_dir + "/recon_w_5en1,trans0231.txt", 'w', encoding="utf_8") as fin:
                fin.write("recon_weight=5e-1, transpose(0,2,3,1), no Normalize((0.1307,), (0.3081,)),")
        self.recon_w = 5e-1
        self.model = None

    def train(self):
        utils.setup_seed(3407)

        # Load data
        transform = transforms.Compose([
            # shift by 2 pixels in either direction with zero padding.
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(root=self.configs["mnist_root"], train=True,
                              transform=transform,
                              download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.configs["batch_size"], shuffle=True)
        test_dataset = MNIST(root=self.configs["mnist_root"], train=False,
                             transform=transform,
                             download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.configs["batch_size"], shuffle=False)

        capsnet = CapsNet()
        capsnet.to(self.device)
        optimizer = optim.Adam(capsnet.parameters(), lr=self.configs["learning_rate"])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        loss_fn = CapsuleLoss(recon_w=self.recon_w).to(self.device)

        # logs = defaultdict(list)
        for epoch_id in range(self.configs["epochs"]):
            # tracker_epoch = defaultdict(lambda: defaultdict(dict))
            correct, total, total_loss = 0., 0., 0.
            for batch_id, (x_img, y_label) in enumerate(tqdm(train_loader, desc="Train|Epoch:")):
                optimizer.zero_grad()

                x_img = x_img.to(self.device)
                y_label = torch.eye(10).index_select(dim=0, index=y_label).to(self.device)

                logits, reconstruction = capsnet(x_img)
                # Compute loss & accuracy
                loss = loss_fn(x_img, y_label, logits, reconstruction)
                correct += torch.sum(
                    torch.argmax(logits, dim=1) == torch.argmax(y_label, dim=1)).item()
                total += len(y_label)
                accuracy = correct / total
                total_loss += loss
                loss.backward()
                optimizer.step()

                # print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(epoch_id + 1,
                #                                                           batch_id,
                #                                                           total_loss / batch_id,
                #                                                           accuracy))

                if batch_id % self.configs["print_every"] == 0 or batch_id == len(train_loader) - 1:
                    print("==> Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}, accuracy: {:.4f}".format(
                        epoch_id, self.configs["epochs"], batch_id, len(train_loader) - 1, total_loss / batch_id, accuracy))

                    z = torch.randn(size=(10, 10, 16)).to(self.device)
                    y_label = torch.zeros(size=(10, 10))
                    labels = torch.arange(0, 10).unsqueeze(1)
                    y_label.scatter_(1, labels, 1)
                    y_label = y_label.to(self.device)
                    # print("one hot:")
                    # print(y_label)

                    recon_x = capsnet.generate(z=z, label=y_label)

                    plt.figure(batch_id)
                    plt.figure(figsize=(5, 10))
                    for p in range(10):
                        plt.subplot(5, 2, p + 1)
                        plt.text(
                            0, 0, "c={:d}".format(p), color='black',
                            backgroundcolor='white', fontsize=8)
                        plt.imshow(recon_x[p].view(28, 28).cpu().data.numpy())
                        plt.axis('off')
                    plt.savefig(self.run_save_dir + "E{:d}_{:d}.png".format(epoch_id, batch_id), dpi=300)
                    plt.clf()
                    plt.close('all')
            scheduler.step()
            print('Total loss for epoch {}: {}'.format(epoch_id + 1, total_loss))
            # if not os.path.exists(self.run_save_dir+"")
            # df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            # g = sns.lmplot(
            #     x='x', y='y', hue='label', data=df.groupby('label').head(100),
            #     fit_reg=False, legend=True)
            # g.savefig(self.run_save_dir + "E{:d}-Dist.png".format(epoch_id), dpi=300)
        capsnet.eval()
        correct, total = 0, 0
        for images, labels in test_loader:
            # Add channels = 1
            images = images.to(self.device)
            # Categogrical encoding
            labels = torch.eye(10).index_select(dim=0, index=labels).to(self.device)
            logits, reconstructions = capsnet(images)
            pred_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
            total += len(labels)
        print('Accuracy: {}'.format(correct / total))

        # Save model
        torch.save(capsnet.state_dict(),
                   self.run_save_dir + 'capsnet_ep{}_acc{}.pt'.format(self.configs["epochs"], correct / total))

    def generate(self, resume="202404081822"):
        if self.model is None:
            self.model = CapsNet()
            state_dict = torch.load(os.path.join(self.run_save_dir+resume, 'capsnet_ep10_acc0.9641.pt'))
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        z = torch.randn(size=(10, 10, 16)).to(self.device)
        y_label = torch.zeros(size=(10, 10))
        labels = torch.arange(0, 10).unsqueeze(1)
        # print(labels)
        y_label.scatter_(1, labels, 1)
        y_label = y_label.to(self.device)
        # print("one hot:")
        # print(y_label)

        recon_x = self.model.generate(z=z, label=y_label)

        plt.figure(0)
        plt.figure(figsize=(5, 10))
        for p in range(10):
            plt.subplot(5, 2, p + 1)
            plt.text(
                0, 0, "c={:d}".format(p), color='black',
                backgroundcolor='white', fontsize=8)
            plt.imshow(recon_x[p].view(28, 28).cpu().data.numpy())
            plt.axis('off')
        plt.savefig(self.run_save_dir+resume + "/A_test1.png", dpi=300)
        plt.clf()
        plt.close('all')


if __name__ == '__main__':
    caps_trainer = TrainerCapsNet(istrain=False)
    caps_trainer.train()
    # caps_trainer.generate()
