#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/14 2:31
# @Author: ZhaoKe
# @File : glow_mnist.py
# @Software: PyCharm
import os
import time
from math import log
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from dlkit.models.flow_glow import Glow
from torchvision.utils import save_image


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []
    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    # n_pixel = image_size * image_size * 3
    n_pixel = image_size * image_size

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def get_argparse():
    parser = argparse.ArgumentParser(description="Glow trainer")
    parser.add_argument("--batch", default=32, type=int, help="batch size")
    parser.add_argument("--iter", default=100000, type=int, help="maximum iterations")
    parser.add_argument(
        "--n_flow", default=4, type=int, help="number of flows in each block"
    )
    parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive"
    )
    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--img_size", default=56, type=int, help="image size")
    parser.add_argument("--temp", default=0.5, type=float, help="temperature of sampling")
    parser.add_argument("--n_sample", default=30, type=int, help="number of samples")
    # parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
    args = parser.parse_args()
    return args


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    dataset = MNIST(root=path, train=True, transform=transform, download=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


class GlowMnistTrainer(object):
    def __init__(self, istrain=True):
        self.args = get_argparse()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_bins = 2.0 ** self.args.n_bits  # 量化
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.in_channels = 1
        # self.demo_test = demo_test
        self.run_save_dir = "../run/glow_mnist/" + self.timestr + '_size56f4b3_affine/'
        if istrain:
            # if not self.demo_test:
            os.makedirs(self.run_save_dir, exist_ok=True)

    def glow_train(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.args.img_size),
                # transforms.CenterCrop(image_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        train_dataset = MNIST(root="F:/DATAS/mnist", train=True, download=True, transform=transform)
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # dataset = iter(sample_data("F:/DATAS/mnist", 32, self.args.img_size))
        z_sample = []
        z_shapes = calc_z_shapes(n_channel=self.in_channels, input_size=self.args.img_size,
                                 n_flow=self.args.n_flow, n_block=self.args.n_block)
        print("z_shape:", z_shapes)
        for z in z_shapes:
            z_new = torch.randn(self.args.n_sample, *z, device=self.device) * self.args.temp
            z_sample.append(z_new)

        model = Glow(in_channel=self.in_channels, n_flow=self.args.n_flow, n_block=self.args.n_block,
                     affine=True, conv_lu=True).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        # input_img = torch.rand(size=(16, 3, self.args.img_size, self.args.img_size), device=self.device)
        # for x_idx in range(20000):
        #     x_img, _ = next(dataset)
        #     x_img = x_img.to(self.device)
        #     x_img = x_img * 255
        #     if self.args.n_bits < 8:
        #         x_img = torch.floor(x_img / 2 ** (8 - self.args.n_bits))
        #     x_img = x_img / self.n_bins - 0.5
        #     optimizer.zero_grad()
        #     if x_idx == 0:
        #         with torch.no_grad():
        #             log_p, logdet, _ = model(
        #                 x_img + torch.rand_like(x_img) / self.n_bins
        #             )
        #             continue
        #     else:
        #         log_p, logdet, _ = model(x_img + torch.rand_like(x_img) / self.n_bins)
        #     logdet = logdet.mean()
        #     loss, log_p, log_det = calc_loss(log_p, logdet, self.args.img_size, self.n_bins)
        #     # model.zero_grad()
        #     loss.backward()
        #     # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
        #     # warmup_lr = self.args.lr
        #     # optimizer.param_groups[0]["lr"] = warmup_lr
        #     optimizer.step()
        #     if x_idx % 50 == 0:
        #         print(f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}")  #; lr: {warmup_lr:.7f}")
        #         # wandb.log({"Loss": loss.item(), "logP": log_p.item(), "logdet": log_det.item(), "lr": warmup_lr})
        #
        #     if x_idx % 1000 == 0:
        #         with torch.no_grad():
        #             generate = model.reverse(z_sample).cpu().data
        #             save_image(
        #                 generate,
        #                 self.run_save_dir+f"{str(x_idx + 1).zfill(6)}.png",
        #                 normalize=True,
        #                 nrow=10,
        #                 range=(-0.5, 0.5),
        #             )
        #             # wandb.log({
        #             #     "images": wandb.Image(generate),  # 接收的是一个numpy格式的数组
        #             # })
        #
        #     if x_idx % 1000 == 0:
        #         torch.save(
        #             model.state_dict(), self.run_save_dir+f"model_{str(x_idx + 1).zfill(6)}.pt"
        #         )
        #         torch.save(
        #             optimizer.state_dict(), self.run_save_dir+f"optim_{str(x_idx + 1).zfill(6)}.pt"
        #         )
        batch_id = 0
        for epoch_id in range(20):
            for x_idx, (x_img, _) in tqdm(enumerate(dataloader), desc="OneEpoch"):
                x_img = x_img * 255
                if self.args.n_bits < 8:
                    x_img = torch.floor(x_img / 2 ** (8 - self.args.n_bits))
                x_img = x_img / self.n_bins - 0.5
                # x_img = F.pad(x_img, [2, 2, 2, 2, 0, 0], mode="constant", value=0.)
                # print("device of x_img", x_img.device)
                # print("shape of input_x:", x_img.shape)
                optimizer.zero_grad()
                if epoch_id == 0 and x_idx == 0:
                    with torch.no_grad():
                        log_p, logdet, _ = model((x_img + torch.rand_like(x_img) / self.n_bins).to(self.device))
                        continue
                else:
                    x_img = (x_img + torch.rand_like(x_img) / self.n_bins).to(self.device)
                    log_p, logdet, _ = model(x_img)

                # print("shape of model output log_p, logdet:", log_p.shape, logdet.shape)
                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(log_p=log_p, logdet=logdet, image_size=self.args.img_size,
                                                 n_bins=self.n_bins)
                # print("loss, log_p, logdet:", loss, log_p, log_det)
                # print("shape of loss loss, log_p, logdet:", loss.shape, log_p.shape, log_det.shape)
                # model.zero_grad()
                loss.backward()
                # warmup_lr = self.args.lr
                # optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()
                if batch_id % 50 == 0:
                    print(f"epoch[{epoch_id}] batch[{x_idx}], loss: {loss.item():.5f}, log_p: {log_p.item():.5f}, log_det: {log_det.item():.5f}")
                    # print(f"lr: {warmup_lr:.7f}")
                if batch_id % 500 == 0:
                    with torch.no_grad():
                        save_image(model.reverse(z_sample).cpu().data,
                                   self.run_save_dir + 'generate_images-e{}b{}.png'.format(epoch_id+1, x_idx + 1),
                                   normalize=True,
                                   nrow=10,
                                   range=(-0.5, 0.5))
                batch_id += 1
            torch.save(model.state_dict(), self.run_save_dir + f"glow_epoch{epoch_id}.pt")
            torch.save(optimizer.state_dict(), self.run_save_dir + f"optim_epoch{epoch_id}.pt")

    def infer(self, resume_model_path):
        model = Glow(in_channel=3, n_flow=self.args.n_flow, n_block=self.args.n_block).to(self.device)
        model.load_state_dict(torch.load(self.run_save_dir+"glow_epoch1.pt"))

        z_sample = None
        with torch.no_grad():
            g_img = model.reverse(z_sample).cpu().data


if __name__ == '__main__':
    # 参考一下 https://github.com/defineHong/glow-mnist/blob/main/app.py

    glow_trainer = GlowMnistTrainer(istrain=True)
    glow_trainer.glow_train()
