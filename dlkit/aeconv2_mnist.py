#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/14 14:49
# @Author: ZhaoKe
# @File : aeconv2_mnist.py
# @Software: PyCharm
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pyro
from pyro.infer import SVI, Trace_ELBO
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from srutils import utils
from dlkit.modules.func import onehot
from dlkit.models.vaeconv_cond import VAE, vae_loss, VAESVI


class TrainerCVAEMNIST(object):
    def __init__(self, istrain=False, istest=True):
        self.istrain, self.istest = istrain, istest
        self.configs = {
            "lr": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "d": 16,
            "fc_input_dim": 128,
            "seed": 3407,
            "epochs": 15,
            "iscond": True,
            "cond_dim": 10,
            "img_size": (1, 28, 28),
            "ismix": False
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.run_save_dir = "../run/vaeconv_mnist/"
        self.kl_weight = 0.0001
        if istrain:
            if self.istest:
                self.run_save_dir += self.timestr + '/'
            else:
                self.run_save_dir += self.timestr + 'ep35la16kl0001cond1mix1/'
                os.makedirs(self.run_save_dir, exist_ok=True)

                with open(self.run_save_dir + "setting_info.txt", 'w', encoding="utf_8") as fin:
                    fin.write("latent_dim={}, epoch={}, kl_weight={}, ".format(self.configs["d"],
                                                                               self.configs["epochs"],
                                                                               self.kl_weight))
                    fin.write("iscond={}, ismix={},".format(self.configs["iscond"], self.configs["ismix"]))
        self.model = None
        self.optim = None
        self.loss_fn = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def set_config(self, key, value):
        self.configs[key] = value

    def __setup_dataset(self):
        data_dir = 'F:/DATAS/mnist'
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        self.test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        train_transform = transforms.Compose([transforms.ToTensor(), ])
        test_transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset.transform = train_transform
        self.test_dataset.transform = test_transform
        if self.configs["iscond"]:
            train_dataset.target_transform = onehot(self.configs["cond_dim"])
            self.test_dataset.target_transform = onehot(self.configs["cond_dim"])

        m = len(train_dataset)
        train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.configs["batch_size"], shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=self.configs["batch_size"], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.configs["batch_size"],
                                                       shuffle=False)
        print("Built Dataset and DataLoader")

    def setup_models(self, model_name="VAE"):
        if model_name == "VAE":
            self.model = VAE(shape=self.configs["img_size"], nhid=self.configs["d"], iscond=self.configs["iscond"],
                             cond_dim=self.configs["cond_dim"])
            self.loss_fn = vae_loss
            if self.istrain:
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.configs["lr"],
                                              weight_decay=self.configs["weight_decay"])
        elif model_name == "VAESVI":
            self.model = VAESVI(shape=self.configs["img_size"], nhid=self.configs["d"], iscond=self.configs["iscond"],
                                cond_dim=self.configs["cond_dim"])
            self.loss_fn = vae_loss
            if self.istrain:
                adam_params = {"lr": self.configs["lr"], "betas": (0.95, 0.999)}
                self.optim = pyro.optim.Adam(adam_params)
        self.model.to(self.device)
        print("Built Model and Optimizer and Loss Function")

    def data_mix(self, x, y):
        mix_rate = int(0.667 * x.shape[0])
        for j in range(x.shape[0]):
            if j < mix_rate:
                r = random.randint(1, 10)
                # print(r, y[j:j+r])
                # print(x[j])
                x[j] = torch.mean(x[j:j + r], dim=0)
                y[j] = torch.sum(y[j:j + r], dim=0)
                # print(x[j])
                # print(r, x[j].shape)
                # print(r, x[j].shape, x[j:j + r].shape, y[j])
                break
        # print(x.shape)
        # print(y.shape)
        return x, y

    def plot_ae_outputs(self, epoch_id):
        plt.figure()
        for i in range(5):
            ax = plt.subplot(2, 5, i + 1)
            img = self.test_dataset[i][0].unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                if self.configs["iscond"]:
                    y = self.test_dataset[i][1].unsqueeze(0).to(self.device)
                    rec_img, _, _ = self.model(img, y)
                else:
                    rec_img, _, _ = self.model(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 5 // 2:
                ax.set_title("Original images")
            ax = plt.subplot(2, 5, i + 1 + 5)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 5 // 2:
                ax.set_title("Reconstructed images")
        plt.savefig(self.run_save_dir + f"test_plot_epoch_{epoch_id}.png", format="png", dpi=300)
        plt.close()

    def train(self):
        """ 无监督，纯AutoEncoder"""
        utils.setup_seed(3407)
        self.__setup_dataset()
        self.setup_models()
        print(self.configs)
        diz_loss = {"train_loss": [], "val_loss": []}
        for epoch in range(self.configs["epochs"]):

            self.model.train()
            train_loss = []
            print(f"Train Epoch {epoch}")
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)

                if self.configs["iscond"]:
                    y = y.to(self.device)
                    if self.configs["ismix"]:
                        mix_x, mix_y = self.data_mix(x, y)
                        decoded_data, latent_mean, latent_logvar = self.model(mix_x, mix_y)
                        loss_value = self.loss_fn(mix_x, decoded_data, latent_mean, latent_logvar,
                                                  kl_weight=self.kl_weight)
                    else:
                        decoded_data, latent_mean, latent_logvar = self.model(x, y)
                        loss_value = self.loss_fn(x, decoded_data, latent_mean, latent_logvar, kl_weight=self.kl_weight)
                else:
                    decoded_data, latent_mean, latent_logvar = self.model(x)
                    loss_value = self.loss_fn(x, decoded_data, latent_mean, latent_logvar, kl_weight=self.kl_weight)

                self.optim.zero_grad()
                loss_value.backward()
                self.optim.step()
                if i % 15 == 0:
                    print(f"\t partial train loss (single batch: {loss_value.data:.6f})")
                train_loss.append(loss_value.detach().cpu().numpy())
            train_loss_value = np.mean(train_loss)

            self.model.eval()
            val_loss = 0.
            with torch.no_grad():
                conc_out = []
                conc_label = []
                conc_mean = []
                conc_logvar = []
                for x, y in self.valid_loader:
                    x = x.to(self.device)
                    if self.configs["iscond"]:
                        y = y.to(self.device)
                        if self.configs["ismix"]:
                            mix_x, mix_y = self.data_mix(x, y)
                            decoded_data, latent_mean, latent_logvar = self.model(mix_x, mix_y)
                        else:
                            decoded_data, latent_mean, latent_logvar = self.model(x, y)
                    else:
                        decoded_data, latent_mean, latent_logvar = self.model(x)
                    conc_out.append(decoded_data.cpu())
                    conc_label.append(x.cpu())
                    conc_mean.append(latent_mean.cpu())
                    conc_logvar.append(latent_logvar.cpu())
                conc_out = torch.cat(conc_out)
                conc_label = torch.cat(conc_label)
                conc_mean = torch.cat(conc_mean)
                conc_logvar = torch.cat(conc_logvar)
                val_loss = self.loss_fn(conc_out, conc_label, conc_mean, conc_logvar, kl_weight=self.kl_weight)
            val_loss_value = val_loss.data
            print(f"\t Epoch {epoch} test loss: {val_loss.item()}")
            diz_loss["train_loss"].append(train_loss_value)
            diz_loss["val_loss"].append(val_loss_value)
            if epoch > 5:
                torch.save(self.model.state_dict(), self.run_save_dir + '{}_epoch_{}.pth'.format("vaeconv", epoch))
            self.plot_ae_outputs(epoch_id=epoch)
        plt.figure()
        plt.semilogy(diz_loss["train_loss"], label="Train")
        plt.semilogy(diz_loss["val_loss"], label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.savefig(self.run_save_dir + "LossIter.png", format="png", dpi=300)
        plt.close()

    def generate(self, resume_path, iscond=True):
        if self.model is None:
            self.setup_models()
        else:
            resume_path = ""
        self.model.eval()
        state_dict = torch.load(os.path.join(self.run_save_dir, resume_path, f'vaeconv_epoch_23.pth'))
        self.model.load_state_dict(state_dict)
        # print(self.model)
        batch_size, class_num = 10, 10
        if not iscond:
            recon_images = self.model.generate(batch_size=batch_size)
        else:
            # z = torch.randn(size=(batch_size, self.configs["d"],), device=self.device)
            # y_label = torch.zeros(size=(batch_size, class_num))
            y_label = torch.ones(size=(batch_size, class_num))
            labels = torch.arange(0, class_num).unsqueeze(1)
            # print(labels)
            y_label.scatter_(1, labels, 1)
            y_label = y_label.to(self.device)
            # y_label[1][7] = 1
            # y_label[0][8] = 1
            # y_label[8][3] = 1
            recon_images = self.model.generate(batch_size, labels=y_label)
        recon_images = recon_images.squeeze().detach().cpu().numpy()
        plt.figure(0)
        for i in range(batch_size):
            plt.subplot(2, 5, i + 1)
            plt.imshow(recon_images[i])
            plt.xticks([])
            plt.yticks([])
            plt.title(f"generate_{i}")
        plt.savefig(self.run_save_dir + resume_path + "/generate_test_1.png", format="png", dpi=300)
        plt.show()

    def train_svi(self):
        """ SVI训练"""
        pyro.clear_param_store()
        utils.setup_seed(3407)
        self.__setup_dataset()
        self.setup_models(model_name="VAESVI")
        print(self.configs)
        self.model.train()
        elbo = Trace_ELBO()
        svi = SVI(self.model.model, self.model.guide, self.optim, loss=elbo)
        # diz_loss = {"train_loss": [], "val_loss": []}
        train_elbo = []
        test_elbo = []
        for epoch in range(self.configs["epochs"]):
            epoch_loss = 0.0
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                epoch_loss += svi.step(x)
            normalizer_train = len(self.train_loader)
            total_epoch_loss_train = epoch_loss / normalizer_train
            train_elbo.append(total_epoch_loss_train)
            print(
                "[epoch %03d]  average training loss: %.4f"
                % (epoch, total_epoch_loss_train)
            )

            if epoch % 1 == 0:
                # initialize loss accumulator
                test_loss = 0.0
                # compute the loss over the entire test set
                for i, (x, _) in enumerate(self.valid_loader):
                    # if on GPU put mini-batch into CUDA memory
                    x = x.cuda()
                    # compute ELBO estimate and accumulate loss
                    test_loss += svi.evaluate_loss(x)

                    # pick three random test images from the first mini-batch and
                    # visualize how well we're reconstructing them
                    if i == 0:
                        # if args.visdom_flag:
                        #     plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = self.model.reconstruct_img(test_img)
                            # vis.image(
                            #     test_img.reshape(28, 28).detach().cpu().numpy(),
                            #     opts={"caption": "test image"},
                            # )
                            # vis.image(
                            #     reco_img.reshape(28, 28).detach().cpu().numpy(),
                            #     opts={"caption": "reconstructed image"},
                            # )
                            plt.figure()
                            plt.subplot(1, 2, 1)
                            plt.imshow(test_img.reshape(28, 28).detach().cpu().numpy())
                            plt.subplot(1, 2, 2)
                            plt.imshow(reco_img.reshape(28, 28).detach().cpu().numpy())
                            plt.savefig(self.run_save_dir + f"test_img_epoch_{epoch}.png", format="png", dpi=300)
                            plt.close()

                # report test diagnostics
                normalizer_test = len(self.valid_loader.dataset)
                total_epoch_loss_test = test_loss / normalizer_test
                test_elbo.append(total_epoch_loss_test)
                print(
                    "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
                )

            if epoch > 5:
                torch.save(self.model.state_dict(), self.run_save_dir + '{}_epoch_{}.pth'.format("vaeconv", epoch))
            self.plot_ae_outputs(epoch_id=epoch)
        plt.figure()
        plt.semilogy(train_elbo, label="Train")
        plt.semilogy(test_elbo, label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.savefig(self.run_save_dir + "LossIter.png", format="png", dpi=300)
        plt.close()
        #     if epoch == args.tsne_iter:
        #         mnist_test_tsne(vae=vae, test_loader=test_loader)
        #         plot_llk(np.array(train_elbo), np.array(test_elbo))
        #
        # return vae


if __name__ == '__main__':
    # trainer = TrainerCVAEMNIST(istrain=True, istest=False)
    # trainer.set_config("iscond", True)
    # trainer.set_config("ismix", True)
    # trainer.train()
    # trainer = TrainerCVAEMNIST(istrain=False, istest=True)
    # trainer.generate(resume_path="202404151323ep35la16kl0001cond", iscond=True)

    trainer = TrainerCVAEMNIST(istrain=True, istest=True)
    trainer.set_config("iscond", False)
    trainer.set_config("ismix", False)
    trainer.set_config("cond_dim", 0)
    trainer.train()
    # trainer.generate(resume_path="202404151323ep35la16kl0001cond", iscond=True)
