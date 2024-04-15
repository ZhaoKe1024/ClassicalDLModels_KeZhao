#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/11 14:08
# @Author: ZhaoKe
# @File : vae_ssl_mnist.py
# @Software: PyCharm
import os
import time
from functools import reduce
from operator import __or__
from itertools import repeat, cycle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm

from dlkit.models.lvae_deepgm import LadderDeepGenerativeModel
from dlkit.modules.func import onehot, log_sum_exp
from dlkit.modules.variational import SVI, DeterministicWarmup, ImportanceWeightedSampler


def binary_cross_entropy(r, x):
    """Drop in replacement until PyTorch adds `reduce` keyword."""
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def flatten_bernoulli(x):
    return transforms.ToTensor()(x).view(-1).bernoulli()


class VAESSLTrainer(object):
    def __init__(self, istrain=True):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA: {}".format(self.cuda))
        self.n_labels = 10
        self.num_workers = 0
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.run_save_dir = "./run/lvae_ssl/"
        if istrain:
            self.run_save_dir += self.timestr + '/'
            os.makedirs(self.run_save_dir, exist_ok=True)

    def get_sampler(self, labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(self.n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(self.n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    def __setup_dataset(self, location="F:/DATAS/mnist", batch_size=64, labels_per_class=100):
        mnist_train = MNIST(location, train=True, download=True,
                            transform=flatten_bernoulli, target_transform=onehot(self.n_labels))
        mnist_valid = MNIST(location, train=False, download=True,
                            transform=flatten_bernoulli, target_transform=onehot(self.n_labels))
        # Dataloaders for MNIST
        labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=self.num_workers,
                                               pin_memory=self.cuda,
                                               sampler=self.get_sampler(mnist_train.train_labels.numpy(),
                                                                        labels_per_class))
        unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=self.num_workers,
                                                 pin_memory=self.cuda,
                                                 sampler=self.get_sampler(mnist_train.train_labels.numpy()))
        validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=self.num_workers,
                                                 pin_memory=self.cuda,
                                                 sampler=self.get_sampler(mnist_valid.test_labels.numpy()))
        return labelled, unlabelled, validation

    def train(self):
        torch.manual_seed(1337)
        np.random.seed(1337)
        labelled, unlabelled, validation = self.__setup_dataset(location="./", batch_size=100, labels_per_class=10)
        print(len(labelled))
        print(len(unlabelled))
        print(len(validation))
        # return
        alpha = 0.1 * len(unlabelled) / len(labelled)

        models = []
        # Kingma 2014, M2 model. Reported: 88%, achieved: ??%
        # from models import DeepGenerativeModel
        # models += [DeepGenerativeModel([784, n_labels, 50, [600, 600]])]

        # Maaløe 2016, ADGM model. Reported: 99.4%, achieved: ??%
        # from models import AuxiliaryDeepGenerativeModel
        # models += [AuxiliaryDeepGenerativeModel([784, n_labels, 100, 100, [500, 500]])]
        models += [LadderDeepGenerativeModel([784, self.n_labels, [32, 16, 8], [128, 128, 128]])]
        for model in models:
            if self.cuda:
                model = model.cuda()
            beta = DeterministicWarmup(n=4 * len(unlabelled) * 100)
            sampler = ImportanceWeightedSampler(mc=1, iw=1)

            elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

            epochs = 251
            best = 0.0

            histry = []
            accu_list = []
            titles = ["train_total_loss",
                      "train_labelled_loss",
                      "train_unlabelled_loss",
                      "test_total_loss",
                      "test_labelled_loss",
                      "test_unlabelled_loss"]
            colors = ["#800000", "#f58231", "#ffe119", "#9A6324", "#000075", "#911eb4"]
            for i in titles:
                histry.append([])
            file = open(self.run_save_dir + "runinfo.log", 'w+')
            for epoch in range(epochs):
                model.train()
                total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                for (x, y), (u, _) in tqdm(zip(cycle(labelled), unlabelled), desc="Train:"):
                    # Wrap in variables
                    x, y, u = Variable(x), Variable(y), Variable(u)

                    if self.cuda:
                        # They need to be on the same device and be synchronized.
                        x, y = x.cuda(device=0), y.cuda(device=0)
                        u = u.cuda(device=0)
                    L, _ = elbo(x, y)
                    U, _ = elbo(u)

                    # Add auxiliary classification loss q(y|x)
                    logits = model.classify(x)
                    classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                    J_alpha = L - alpha * classication_loss + U
                    # print(f"J_alpha, L, U: {J_alpha}, {L}, {U}")
                    J_alpha.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += J_alpha.data.item()
                    labelled_loss += L.data.item()
                    unlabelled_loss += U.data.item()
                m = len(unlabelled)
                histry[0].append(total_loss / m)
                histry[1].append(labelled_loss / m)
                histry[2].append(unlabelled_loss / m)
                print(*(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m), sep="\t", file=file)
                if epoch % 1 == 0:
                    model.eval()
                    print("Epoch: {}".format(epoch))
                    print("[Train]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                                 labelled_loss / m,
                                                                                                 unlabelled_loss / m,
                                                                                                 accuracy / m))
                    total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                    for x, y in validation:
                        x, y = Variable(x), Variable(y)

                        if self.cuda:
                            x, y = x.cuda(device=0), y.cuda(device=0)

                        L, _ = elbo(x, y)
                        U, _ = elbo(x)

                        logits = model.classify(x)
                        classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                        J_alpha = L + alpha * classication_loss + U

                        total_loss += J_alpha.data.item()
                        labelled_loss += L.data.item()
                        unlabelled_loss += U.data.item()

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy += torch.mean((pred_idx.data == lab_idx.data).float())
                    m = len(validation)
                    histry[3].append(total_loss / m)
                    histry[4].append(labelled_loss / m)
                    histry[5].append(unlabelled_loss / m)
                    # print(accuracy)
                    accu_list.append(accuracy.data.item() / m)
                    m = len(validation)
                    print(*(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m), sep="\t",
                          file=file)
                    print(
                        "[Validation] J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                                  labelled_loss / m,
                                                                                                  unlabelled_loss / m,
                                                                                                  accuracy / m))
                if epoch > 30 or epoch < 2:
                    ax = plt.figure(epoch)
                    plts = []
                    for i in range(len(histry)):
                        p, = plt.plot(range(len(histry[i])), histry[i], c=colors[i])
                        plts.append(p)
                    plt.legend(handles=plts, labels=titles, loc='best')
                    plt.xlabel("iter")
                    plt.ylabel("accuracy")
                    plt.savefig(self.run_save_dir + f"train_test_loss_{epoch}.png", dpi=300, format='png')
                    plt.close()

                    plt.figure(epoch)
                    plt.plot(range(len(accu_list)), accu_list, c="#000000")
                    plt.xlabel("iter")
                    plt.ylabel("test accuracy")
                    plt.savefig(self.run_save_dir + f"test_accuracy_{epoch}.png", dpi=300, format='png')
                    plt.close()

                if accuracy > best:
                    best = accuracy
                    torch.save(model, self.run_save_dir + '{}_epoch_{}_accu_{}.pt'.format("lvaessl", epoch,
                                                                                          str(accuracy * 10000)[:4]))

    def reconstruct(self, location="F:/DATAS/mnist", batch_size=64, resume_path="202404121748/"):
        self.run_save_dir += resume_path
        mnist_valid = MNIST(location, train=False, download=True,
                            transform=flatten_bernoulli, target_transform=onehot(self.n_labels))

        validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=self.num_workers,
                                                 pin_memory=self.cuda,
                                                 sampler=self.get_sampler(mnist_valid.test_labels.numpy()))
        # model = LadderDeepGenerativeModel([784, self.n_labels, [32, 16, 8], [128, 128, 128]])
        model = torch.load(self.run_save_dir + "lvaessl_epoch_75_accu_tens.pt")
        # print(state_dict)
        # model.load_state_dict(state_dict)
        if self.cuda:
            model = model.cuda()
        model.eval()
        beta = DeterministicWarmup(n=4 * len(validation) * 100)
        sampler = ImportanceWeightedSampler(mc=1, iw=1)
        elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
        cnt = 4
        for i, (x, y) in enumerate(validation):
            print(x.shape, y.shape)
            x, y = Variable(x), Variable(y)
            if self.cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)
            # print(elbo(x, y))
            _, recon_labelled = elbo(x, y)
            _, recon_unlabelled = elbo(x)
            print(recon_labelled.shape, recon_unlabelled.shape)
            for j in range(4):
                plt.figure(j)

                plt.subplot(1, 3, 1)
                plt.imshow(x[j].reshape(28, 28).detach().cpu().numpy())
                plt.subplot(1, 3, 2)
                plt.imshow(recon_labelled[j].reshape(28, 28).detach().cpu().numpy())
                plt.subplot(1, 3, 3)
                plt.imshow(recon_unlabelled[j].reshape(28, 28).detach().cpu().numpy())

                plt.savefig(self.run_save_dir + f"recon_batch_{j}.png", dpi=300, format="png")
            break
