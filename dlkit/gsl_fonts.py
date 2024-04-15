#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/12 18:28
# @Author: ZhaoKe
# @File : gsl_fonts.py
# @Software: PyCharm
import itertools
import os
import numpy as np
import visdom
import yaml
import time
import functools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from dlkit.data_utils.fontv1_reader import return_data
from dlkit.modules.func import setup_seed, cuda
from dlkit.modules.gsl_modules import Generator_fc, get_norm_layer, DataGather
from gsl_train import get_args


class TrainerGSL(object):
    def __init__(self, configs="../configs/gan.yaml", args=get_args(), istrain=True):
        self.configs = None
        with open(configs, 'r', encoding='utf-8') as stream:
            self.configs = yaml.safe_load(stream)
        setup_seed(3407)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.is_train = istrain
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_start_0/'
            if istrain:
                os.makedirs(self.run_save_dir, exist_ok=True)
        self.args = get_args()
        self.use_cuda = args.cuda and torch.cuda.is_available()

        if args.dataset.lower() == "fonts":
            self.nc = 3
            self.z_dim = 100  # 'dimension of the latent representation z'
            # content(letter): 0~20; size: 20~40; font_color: 40~60; back_color: 60~80; style(font): 20
            self.z_content_dim = 20  # 'dimension of the z_content (letter) latent representation in z'
            self.z_size_dim = 20  # 'dimension of the z_size latent representation in z'
            self.z_font_color_dim = 20  # 'dimension of the z_font_color latent representation in z'
            self.z_back_color_dim = 20  # 'dimension of the z_back_color latent representation in z'
            self.z_style_dim = 20  # 'dimension of the z_style latent representation in z'

            self.z_content_start_dim = 0
            self.z_size_start_dim = 20
            self.z_font_color_start_dim = 40
            self.z_back_color_start_dim = 60
            self.z_style_start_dim = 80

        self.dataset = args.dataset
        if args.train:  # train mode
            self.train = True
        else:  # test mode
            self.train = False
            args.batch_size = 1
            self.pretrain_model_path = args.pretrain_model_path
            self.test_img_path = args.test_img_path
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)  ### key

        # model training param
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.norm_layer = get_norm_layer(norm_type=args.norm)
        self.max_iter = args.max_iter
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup
        if args.dataset.lower() == 'dsprites':
            pass
        else:
            self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        self.Autoencoder.to(self.device)
        self.auto_optim = optim.Adam(self.Autoencoder.parameters(), lr=self.lr,
                                     betas=(self.beta1, self.beta2))

        # log and save
        self.log_dir = '../run/GSLAE/checkpoints/' + args.viz_name
        self.model_save_dir = args.model_save_dir
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_combine_sup = None
        self.win_combine_unsup = None

        self.gather_step = args.gather_step
        self.gather = DataGather()
        self.display_step = args.display_step
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.resume_iters = args.resume_iters
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_step = args.save_step
        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def restore_model(self, resume_iters):
        """Restore the trained generator"""
        if resume_iters == 'pretrained':
            print('Loading the pretrained models from  {}...'.format(self.pretrain_model_path))
            self.Autoencoder.load_state_dict(
                torch.load(self.pretrain_model_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} '".format(self.pretrain_model_path))
        else:  # not test
            print('Loading the trained models from step {}...'.format(resume_iters))
            Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
            self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))

    def train_font(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:

                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                E_img = sup_package['E']
                F_img = sup_package['F']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                E_img = Variable(cuda(E_img, self.use_cuda))
                F_img = Variable(cuda(F_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)
                E_recon, E_z = self.Autoencoder(E_img)
                F_recon, F_z = self.Autoencoder(F_img)
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

                A_z_1 = A_z[:, 0:self.z_size_start_dim]  # 0-20
                A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                A_z_3 = A_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                A_z_4 = A_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                A_z_5 = A_z[:, self.z_style_start_dim:]  # 80-100
                B_z_1 = B_z[:, 0:self.z_size_start_dim]  # 0-20
                B_z_2 = B_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                B_z_4 = B_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                B_z_5 = B_z[:, self.z_style_start_dim:]  # 80-100
                C_z_1 = C_z[:, 0:self.z_size_start_dim]  # 0-20
                C_z_2 = C_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                C_z_3 = C_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                C_z_4 = C_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                C_z_5 = C_z[:, self.z_style_start_dim:]  # 80-100
                D_z_1 = D_z[:, 0:self.z_size_start_dim]  # 0-20
                D_z_2 = D_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                D_z_3 = D_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                D_z_5 = D_z[:, self.z_style_start_dim:]  # 80-100
                E_z_1 = E_z[:, 0:self.z_size_start_dim]  # 0-20
                E_z_2 = E_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                E_z_3 = E_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                E_z_4 = E_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
                F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20
                F_z_2 = F_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                F_z_3 = F_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                F_z_4 = F_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                F_z_5 = F_z[:, self.z_style_start_dim:]  # 80-100

                ## 2. combine with strong supervise
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
                # C A same content 1
                A1Co_combine_2C = torch.cat((A_z_1, C_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_A1Co = self.Autoencoder.fc_decoder(A1Co_combine_2C)
                mid_A1Co = mid_A1Co.view(A1Co_combine_2C.shape[0], 256, 8, 8)
                A1Co_2C = self.Autoencoder.decoder(mid_A1Co)

                AoC1_combine_2A = torch.cat((C_z_1, A_z_2, A_z_3, A_z_4, A_z_5), dim=1)
                mid_AoC1 = self.Autoencoder.fc_decoder(AoC1_combine_2A)
                mid_AoC1 = mid_AoC1.view(AoC1_combine_2A.shape[0], 256, 8, 8)
                AoC1_2A = self.Autoencoder.decoder(mid_AoC1)

                # C B same size 2
                B2Co_combine_2C = torch.cat((C_z_1, B_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_B2Co = self.Autoencoder.fc_decoder(B2Co_combine_2C)
                mid_B2Co = mid_B2Co.view(B2Co_combine_2C.shape[0], 256, 8, 8)
                B2Co_2C = self.Autoencoder.decoder(mid_B2Co)

                BoC2_combine_2B = torch.cat((B_z_1, C_z_2, B_z_3, B_z_4, B_z_5), dim=1)
                mid_BoC2 = self.Autoencoder.fc_decoder(BoC2_combine_2B)
                mid_BoC2 = mid_BoC2.view(BoC2_combine_2B.shape[0], 256, 8, 8)
                BoC2_2B = self.Autoencoder.decoder(mid_BoC2)

                # C D same font_color 3
                D3Co_combine_2C = torch.cat((C_z_1, C_z_2, D_z_3, C_z_4, C_z_5), dim=1)
                mid_D3Co = self.Autoencoder.fc_decoder(D3Co_combine_2C)
                mid_D3Co = mid_D3Co.view(D3Co_combine_2C.shape[0], 256, 8, 8)
                D3Co_2C = self.Autoencoder.decoder(mid_D3Co)

                DoC3_combine_2D = torch.cat((D_z_1, D_z_2, C_z_3, D_z_4, D_z_5), dim=1)
                mid_DoC3 = self.Autoencoder.fc_decoder(DoC3_combine_2D)
                mid_DoC3 = mid_DoC3.view(DoC3_combine_2D.shape[0], 256, 8, 8)
                DoC3_2D = self.Autoencoder.decoder(mid_DoC3)

                # C E same back_color 4
                E4Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, E_z_4, C_z_5), dim=1)
                mid_E4Co = self.Autoencoder.fc_decoder(E4Co_combine_2C)
                mid_E4Co = mid_E4Co.view(E4Co_combine_2C.shape[0], 256, 8, 8)
                E4Co_2C = self.Autoencoder.decoder(mid_E4Co)

                EoC4_combine_2E = torch.cat((E_z_1, E_z_2, E_z_3, C_z_4, E_z_5), dim=1)
                mid_EoC4 = self.Autoencoder.fc_decoder(EoC4_combine_2E)
                mid_EoC4 = mid_EoC4.view(EoC4_combine_2E.shape[0], 256, 8, 8)
                EoC4_2E = self.Autoencoder.decoder(mid_EoC4)

                # C F same style 5
                F5Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, C_z_4, F_z_5), dim=1)
                mid_F5Co = self.Autoencoder.fc_decoder(F5Co_combine_2C)
                mid_F5Co = mid_F5Co.view(F5Co_combine_2C.shape[0], 256, 8, 8)
                F5Co_2C = self.Autoencoder.decoder(mid_F5Co)

                FoC5_combine_2F = torch.cat((F_z_1, F_z_2, F_z_3, F_z_4, C_z_5), dim=1)
                mid_FoC5 = self.Autoencoder.fc_decoder(FoC5_combine_2F)
                mid_FoC5 = mid_FoC5.view(FoC5_combine_2F.shape[0], 256, 8, 8)
                FoC5_2F = self.Autoencoder.decoder(mid_FoC5)

                # combine_2C
                A1B2D3E4F5_combine_2C = torch.cat((A_z_1, B_z_2, D_z_3, E_z_4, F_z_5), dim=1)
                mid_A1B2D3E4F5 = self.Autoencoder.fc_decoder(A1B2D3E4F5_combine_2C)
                mid_A1B2D3E4F5 = mid_A1B2D3E4F5.view(A1B2D3E4F5_combine_2C.shape[0], 256, 8, 8)
                A1B2D3E4F5_2C = self.Autoencoder.decoder(mid_A1B2D3E4F5)

                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
                mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
                mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 8, 8)
                A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)

                '''
                optimize for autoencoder
                '''
                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                E_recon_loss = torch.mean(torch.abs(E_img - E_recon))
                F_recon_loss = torch.mean(torch.abs(F_img - F_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

                # 2. sup_combine_loss
                A1Co_2C_loss = torch.mean(torch.abs(C_img - A1Co_2C))
                AoC1_2A_loss = torch.mean(torch.abs(A_img - AoC1_2A))
                B2Co_2C_loss = torch.mean(torch.abs(C_img - B2Co_2C))
                BoC2_2B_loss = torch.mean(torch.abs(B_img - BoC2_2B))
                D3Co_2C_loss = torch.mean(torch.abs(C_img - D3Co_2C))
                DoC3_2D_loss = torch.mean(torch.abs(D_img - DoC3_2D))
                E4Co_2C_loss = torch.mean(torch.abs(C_img - E4Co_2C))
                EoC4_2E_loss = torch.mean(torch.abs(E_img - EoC4_2E))
                F5Co_2C_loss = torch.mean(torch.abs(C_img - F5Co_2C))
                FoC5_2F_loss = torch.mean(torch.abs(F_img - FoC5_2F))
                A1B2D3E4F5_2C_loss = torch.mean(torch.abs(C_img - A1B2D3E4F5_2C))
                combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

                # 3. unsup_combine_loss
                _, A2B3D4E5F1_z = self.Autoencoder(A2B3D4E5F1_2N)
                combine_unsup_loss = torch.mean(
                    torch.abs(F_z_1 - A2B3D4E5F1_z[:, 0:self.z_size_start_dim])) + torch.mean(
                    torch.abs(A_z_2 - A2B3D4E5F1_z[:, self.z_size_start_dim: self.z_font_color_start_dim])) \
                                     + torch.mean(
                    torch.abs(B_z_3 - A2B3D4E5F1_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim])) \
                                     + torch.mean(
                    torch.abs(D_z_4 - A2B3D4E5F1_z[:, self.z_back_color_start_dim: self.z_style_start_dim])) \
                                     + torch.mean(torch.abs(E_z_5 - A2B3D4E5F1_z[:, self.z_style_start_dim:]))

                # whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                # 　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                    self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])
                f.close()

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter, recon_loss=recon_loss.data,
                                       combine_sup_loss=combine_sup_loss.data,
                                       combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter % self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=E_img.data)
                        self.gather.insert(images=F_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.fonts_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoC1_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoC2_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(D3Co_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoC3_2D).data)
                        self.gather.insert(combine_supimages=F.sigmoid(EoC4_2E).data)
                        self.gather.insert(combine_supimages=F.sigmoid(FoC5_2F).data)
                        self.fonts_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(A1B2D3E4F5_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(A2B3D4E5F1_2N).data)
                        self.fonts_viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter % self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name,
                                             '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def test_fonts(self):
        # self.net_mode(train=True)
        # load pretrained model
        self.restore_model('pretrained')
        for index, sup_package in enumerate(self.data_loader):
            A_img = sup_package['A']
            B_img = sup_package['B']
            D_img = sup_package['D']
            E_img = sup_package['E']
            F_img = sup_package['F']

            A_img = Variable(cuda(A_img, self.use_cuda))
            B_img = Variable(cuda(B_img, self.use_cuda))
            D_img = Variable(cuda(D_img, self.use_cuda))
            E_img = Variable(cuda(E_img, self.use_cuda))
            F_img = Variable(cuda(F_img, self.use_cuda))

            ## 1. A B C seperate(first400: id last600 background)
            A_recon, A_z = self.Autoencoder(A_img)
            B_recon, B_z = self.Autoencoder(B_img)
            D_recon, D_z = self.Autoencoder(D_img)
            E_recon, E_z = self.Autoencoder(E_img)
            F_recon, F_z = self.Autoencoder(F_img)
            ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

            A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
            F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20

            # '''  need unsupervise '''
            A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
            mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
            mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 8, 8)
            A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)

            # save synthesized image
            self.test_iter = index
            self.gather.insert(test=F.sigmoid(A2B3D4E5F1_2N).data)
            self.fonts_viz_test()
            self.gather.flush()

    def fonts_save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_ori_E = image[4].squeeze(0)
            image_ori_F = image[5].squeeze(0)
            image_recon = image[6].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_ori_E = unloader(image_ori_E)
            image_ori_F = unloader(image_ori_F)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_ori_E.save(os.path.join(dir, '{}-E_img.png'.format(self.global_iter)))
            image_ori_F.save(os.path.join(dir, '{}-F_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':

            image_AoC1_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoC2_2B = image[1].squeeze(0)
            image_D3Co_2C = image[2].squeeze(0)
            image_DoC3_2D = image[3].squeeze(0)
            image_EoC4_2E = image[4].squeeze(0)
            image_FoC5_2F = image[5].squeeze(0)

            image_AoC1_2A = unloader(image_AoC1_2A)
            image_BoC2_2B = unloader(image_BoC2_2B)
            image_D3Co_2C = unloader(image_D3Co_2C)
            image_DoC3_2D = unloader(image_DoC3_2D)
            image_EoC4_2E = unloader(image_EoC4_2E)
            image_FoC5_2F = unloader(image_FoC5_2F)

            image_AoC1_2A.save(os.path.join(dir, '{}-AoC1_2A.png'.format(self.global_iter)))
            image_BoC2_2B.save(os.path.join(dir, '{}-BoC2_2B.png'.format(self.global_iter)))
            image_D3Co_2C.save(os.path.join(dir, '{}-D3Co_2C.png'.format(self.global_iter)))
            image_DoC3_2D.save(os.path.join(dir, '{}-DoC3_2D.png'.format(self.global_iter)))
            image_EoC4_2E.save(os.path.join(dir, '{}-EoC4_2E.png'.format(self.global_iter)))
            image_FoC5_2F.save(os.path.join(dir, '{}-FoC5_2F.png'.format(self.global_iter)))

        elif mode == 'combine_unsup':
            image_A1B2D3E4F5_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_A2B3D4E5F1_2N = image[1].squeeze(0)

            image_A1B2D3E4F5_2C = unloader(image_A1B2D3E4F5_2C)
            image_A2B3D4E5F1_2N = unloader(image_A2B3D4E5F1_2N)

            image_A1B2D3E4F5_2C.save(os.path.join(dir, '{}-A1B2D3E4F5_2C.png'.format(self.global_iter)))
            image_A2B3D4E5F1_2N.save(os.path.join(dir, '{}-A2B3D4E5F1_2N.png'.format(self.global_iter)))
        elif mode == 'test':
            image_A2B3D4E5F1_2N = image
            image_A2B3D4E5F1_2N = unloader(image_A2B3D4E5F1_2N)
            image_A2B3D4E5F1_2N.save(
                os.path.join(self.output_dir, 'group{}-image_A2B3D4E5F1_2N.png'.format(self.test_iter + 1)))

    def fonts_viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_E = self.gather.data['images'][4][:100]
        x_E = make_grid(x_E, normalize=True)
        x_F = self.gather.data['images'][5][:100]
        x_F = make_grid(x_F, normalize=True)
        x_A_recon = self.gather.data['images'][6][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_E, x_F, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + '_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'recon')
        # self.net_mode(train=True)

    def fonts_viz_combine_recon(self):
        # self.net_mode(train=False)
        AoC1_2A = self.gather.data['combine_supimages'][0][:100]
        AoC1_2A = make_grid(AoC1_2A, normalize=True)
        BoC2_2B = self.gather.data['combine_supimages'][1][:100]
        BoC2_2B = make_grid(BoC2_2B, normalize=True)
        D3Co_2C = self.gather.data['combine_supimages'][2][:100]
        D3Co_2C = make_grid(D3Co_2C, normalize=True)
        DoC3_2D = self.gather.data['combine_supimages'][3][:100]
        DoC3_2D = make_grid(DoC3_2D, normalize=True)
        EoC4_2E = self.gather.data['combine_supimages'][4][:100]
        EoC4_2E = make_grid(EoC4_2E, normalize=True)
        FoC5_2F = self.gather.data['combine_supimages'][5][:100]
        FoC5_2F = make_grid(FoC5_2F, normalize=True)
        images = torch.stack([AoC1_2A, BoC2_2B, D3Co_2C, DoC3_2D, EoC4_2E, FoC5_2F], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + 'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'combine_sup')

    def fonts_viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        A1B2D3E4F5_2C = self.gather.data['combine_unsupimages'][0][:100]
        A1B2D3E4F5_2C = make_grid(A1B2D3E4F5_2C, normalize=True)
        A2B3D4E5F1_2N = self.gather.data['combine_unsupimages'][1][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = torch.stack([A1B2D3E4F5_2C, A2B3D4E5F1_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + 'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'combine_unsup')

    def fonts_viz_test(self):
        # self.net_mode(train=False)
        A2B3D4E5F1_2N = self.gather.data['test'][0][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = A2B3D4E5F1_2N
        self.fonts_save_sample_img(images, 'test')

    def viz_lines(self):
        # self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        combine_sup_loss = torch.stack(self.gather.data['combine_sup_loss']).cpu()
        combine_unsup_loss = torch.stack(self.gather.data['combine_unsup_loss']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                X=iters,
                Y=recon_losses,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='reconsturction loss', ))
        else:
            self.win_recon = self.viz.line(
                X=iters,
                Y=recon_losses,
                env=self.viz_name + '_lines',
                win=self.win_recon,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='reconsturction loss', ))

        if self.win_combine_sup is None:
            self.win_combine_sup = self.viz.line(
                X=iters,
                Y=combine_sup_loss,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend[:self.z_dim],
                    xlabel='iteration',
                    title='combine_sup_loss', ))
        else:
            self.win_combine_sup = self.viz.line(
                X=iters,
                Y=combine_sup_loss,
                env=self.viz_name + '_lines',
                win=self.win_combine_sup,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend[:self.z_dim],
                    xlabel='iteration',
                    title='combine_sup_loss', ))

        if self.win_combine_unsup is None:
            self.win_combine_unsup = self.viz.line(
                X=iters,
                Y=combine_unsup_loss,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend[:self.z_dim],
                    xlabel='iteration',
                    title='combine_unsup_loss', ))
        else:
            self.win_combine_unsup = self.viz.line(
                X=iters,
                Y=combine_sup_loss,
                env=self.viz_name + '_lines',
                win=self.win_combine_unsup,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend[:self.z_dim],
                    xlabel='iteration',
                    title='combine_unsup_loss', ))
