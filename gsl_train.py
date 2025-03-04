#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/12 18:38
# @Author: ZhaoKe
# @File : gsl_train.py
# @Software: PyCharm
import argparse
from dlkit.utils.utils import str2bool
import numpy as np
import torch


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = Solver(args)
    if args.train:
        if args.dataset.lower() == "fonts":
            solver.train_fonts()
        else:
            pass
    else:
        if args.dataset.lower() == 'fonts':
            solver.test_fonts()


def get_args():
    parser = argparse.ArgumentParser(description='GZS-Net')
    '''
    Need modify base on your dataset
    '''
    parser.add_argument('--train', default=False, type=str2bool, help='train: True or test: False')
    parser.add_argument('--dataset', default='ilab_20M', type=str,
                        help='dataset name:[ilab_20M, Fonts, RaFD, dsprites, ilab_20M_custom]')
    parser.add_argument('--dataset_path', default='/home2/andy/ilab2M_pose/train_img_c00_10class', type=str,
                        help='dataset path')
    parser.add_argument('--resume_iters', type=int, default=0,
                        help='resume training from this step, e,g, 10000 or test model selection')
    parser.add_argument('--viz_name', default='ilab_20M', type=str, help='visdom env name')
    parser.add_argument('--image_size', type=int, default=128, help='crop size for the ilab dataset')
    parser.add_argument('--pretrain_model_path', default='./run/GSL/checkpoints/pretrained_models/ilab_20M.ckpt', type=str,
                        help='pretrain model path')

    parser.add_argument('--test_img_path', default='./run/GSL/checkpoints/test_imgs/', type=str,
                        help='pretrain model path')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e7, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    # model params
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=1,
                        help='number of residual blocks in G for encoder and decoder')
    parser.add_argument('--lambda_combine', type=float, default=1, help='weight for lambda_combine')
    parser.add_argument('--lambda_unsup', default=0, type=float, help='lambda_recon')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--num_workers', default=32, type=int, help='dataloader num_workers')
    # log
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    # save model
    # parser.add_argument('--model_save_dir', default='checkpoints', type=str, help='output directory')
    parser.add_argument('--model_save_dir', default='/run/GSL/Group_Supervised_Learning', type=str,
                        help='output directory')

    parser.add_argument('--gather_step', default=5000, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=5000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=10000, type=int,
                        help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    from dlkit.data_utils.fontv1_reader import return_data
    fonts_loader = return_data(args=args)

    # main(args)
