#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/6 13:55
# @Author: ZhaoKe
# @File : Trainer_AlexNet.py
# @Software: PyCharm
import json
import os
import shutil
import time
from datetime import timedelta

import torchvision.transforms
import yaml
import torch
import platform
import numpy as np
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DistributedSampler, DataLoader
from torchinfo import summary
from torchvision.transforms import transforms
from tqdm import tqdm
from visualdl import LogWriter

from cls import SUPPORT_MODEL, __version__
from cls.metric.metrics import accuracy
from cls.models.AlexNet import AlexNet
from cls.utils.logger import setup_logger
from cls.utils.scheduler import WarmupCosineSchedulerLR
from cls.utils.utils import print_arguments, dict_to_object, plot_confusion_matrix
from cls.data_utils.CatdogReader import CatdogDataset as MyDataset

logger = setup_logger(__name__)


class AlexNetTrainer(object):
    def __init__(self, configs, use_gpu=True):
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu

        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.model = None
        self.valid_loader = None
        self.amp_scaler = None

        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]

        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

    # 固定操作，局部小改，还可以再固定
    def __setup_dataloader(self, is_train=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            self.train_dataset = MyDataset(root="E:/DATAS/catdogclass/train",
                                           list_file=self.configs.dataset_conf.train_list,
                                            train=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))
            # 设置支持多卡训练
            train_sampler = None
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=None,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           **self.configs.dataset_conf.dataLoader)
        # 获取测试数据
        self.valid_dataset = MyDataset(root="E:/DATAS/catdogclass/val",
                                       list_file=self.configs.dataset_conf.test_list,
                                        train=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       collate_fn=None,
                                       shuffle=True,
                                       batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                       num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    def __setup_model(self, is_train=False):
        # 自动获取列表数量
        if self.configs.model_conf.num_class is None:
            print("label nums:", self.class_labels, ", length:", len(self.class_labels))
            self.configs.model_conf.num_class = len(self.class_labels)
        # 获取模型
        if self.configs.use_model == 'AlexNet':
            self.model = AlexNet(num_classes=self.configs.model_conf.num_class)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        self.model.to(self.device)
        # summary(self.model, input_size=(1, 98, self.audio_featurizer.feature_dim))
        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        # print(self.model)
        weight = None
        if self.configs.train_conf.loss_weight is not None:
            weight = torch.tensor(self.configs.train_conf.loss_weight, dtype=torch.float, device=self.device)
        else:
            weight = None
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=self.configs.optimizer_conf.learning_rate,
                                                  weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=self.configs.optimizer_conf.learning_rate,
                                                   weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                                 lr=self.configs.optimizer_conf.learning_rate,
                                                 weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer}')
            scheduler_args = {}
            if self.configs.optimizer_conf.get('scheduler_args', {}) is not None:
                # 学习率衰减函数
                scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {})
            else:
                scheduler_args = {}
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                   T_max=max_step,
                                                   **scheduler_args)
            elif self.configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
                self.scheduler = WarmupCosineSchedulerLR(optimizer=self.optimizer,
                                                         fix_epoch=self.configs.train_conf.max_epoch,
                                                         step_per_epoch=len(self.train_loader),
                                                         **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')

    # 纯纯的固定操作，看了下完全可以单独摘出去
    def __load_pretrained(self, pretrained_model):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_dict = self.model.module.state_dict()
            else:
                model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)
            # 过滤不存在的参数
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

    # 固定操作，可以摘出去
    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pth'))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))
            # 自动混合精度参数
            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
                self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
            self.optimizer.step()
            [self.scheduler.step() for _ in range(last_epoch * len(self.train_loader))]
        return last_epoch, best_acc

    # 固定操作
    def __save_checkpoint(self, save_model_path, epoch_id, best_acc=0, best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(state_dict, os.path.join(model_path, 'model.pth'))
        # 自动混合精度参数
        if self.amp_scaler is not None:
            torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "accuracy": best_acc, "version": __version__}
            f.write(json.dumps(data))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * self.configs.train_conf.max_epoch
        for batch_id, (imgs, label) in enumerate(self.train_loader):
            # Step 4 to and forward
            if nranks > 1:
                imgs = imgs.to(local_rank)
                label = label.to(local_rank).long()
            else:
                imgs = imgs.to(self.device)
                label = label.to(self.device).long()

            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                output = self.model(imgs)

            # Step 5 calculate loss function value
            # 计算损失值
            label_tmp = np.zeros((len(label.cpu().numpy()), self.configs.model_conf.num_class))
            # print(label_tmp.shape, label.shape)

            for rid in range(len(label_tmp)):
                label_tmp[rid][label[rid]-1] = 1
            los = self.loss(output, torch.tensor(label_tmp, dtype=torch.float32).to(self.device))

            # Step 6 backward
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()

            # Step 7 optim update
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()

            # Step 8 optim zero_grad
            self.optimizer.zero_grad()

            # 计算准确率
            acc = accuracy(output, label)
            accuracies.append(acc)
            loss_sum.append(los.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), self.train_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_step += 1
            start = time.time()
            self.scheduler.step()

    def train(self, save_model_path="models/", resume_model=None, pretrained_model=None):
        # 获取有多少张显卡训练
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化NCCL环境
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])
        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(is_train=True)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_acc = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                logger.info('=' * 70)
                loss, acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), loss, acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', acc, test_step)
                writer.add_scalar('Test/Loss', loss, test_step)
                test_step += 1
                self.model.train()
                # # 保存最优模型
                if acc >= best_acc:
                    best_acc = acc
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc,
                                           best_model=True)
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc)

    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
                评估模型
                :param resume_model: 所使用的模型
                :param save_matrix_path: 保存混合矩阵的路径
                :return: 评估结果
                """
        if self.valid_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model()
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (img, label) in enumerate(tqdm(self.valid_loader)):
                # Step 4 forward
                img = img.to(self.device)
                label = label.to(self.device).long()

                output = eval_model(img)

                label_tmp = np.zeros((len(label.cpu().numpy()), self.configs.model_conf.num_class))
                # print(label_tmp.shape, label.shape)

                for rid in range(len(label_tmp)):
                    label_tmp[rid][label[rid] - 1] = 1
                los = self.loss(output, torch.tensor(label_tmp, dtype=torch.float32).to(self.device))
                # 计算准确率
                acc = accuracy(output, label)
                accuracies.append(acc)
                # 模型预测标签
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        # 保存混合矩阵
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                  class_labels=self.class_labels)

        self.model.train()
        return loss, acc

    def test(self):
        pass

    # 固定操作
    def export(self, save_model_path="models/", resume_model="models/alexnet/best_model/"):
        """
                导出预测模型
                :param save_model_path: 模型保存的路径
                :param resume_model: 准备转换的模型路径
                :return:
                """
        self.__setup_model()
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
