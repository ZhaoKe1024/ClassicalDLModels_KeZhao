#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/12 16:02
# @Author: ZhaoKe
# @File : train_vae_ssl.py
# @Software: PyCharm
from dlkit.vae_ssl_mnist import VAESSLTrainer


def main():
    # trainer_ssl = VAESSLTrainer(istrain=True)
    # trainer_ssl.train()
    trainer_ssl = VAESSLTrainer(istrain=False)
    trainer_ssl.reconstruct(resume_path="202404121940/")


if __name__ == '__main__':
    main()
