import sys
import argparse
import yaml
import math
import numpy as np
import wandb

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from net.sim_loss import CosineSimLoss

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PT_Processor(Processor):
    """
        Processor for Pretraining.
    """

    def __init__(self, argv=None):
        super().__init__(argv)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        print(self.model)
        self.loss = nn.CrossEntropyLoss()
        self.re_criterion = torch.nn.L1Loss(reduction='none')
        self.sim_loss = CosineSimLoss()
        if not self.disable_wandb:
            wandb.init(project="aimclr", group="dev", config=self.arg)
            wandb.watch(self.model)

    def load_lr(self):
        self.arg.base_lr = self.arg.base_lr * (self.arg.batch_size / 512)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def load_scheduler(self):
        if self.arg.lr_scheduler == 'step' and self.arg.step:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.arg.step, gamma=0.1)
        elif self.arg.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.arg.base_lr, eta_min=self.arg.base_lr//100.)
        else:
            raise ValueError()
        self.lr = self.arg.base_lr

    def adjust_lr(self):
        self.scheduler.step()
        self.lr = self.scheduler.get_lr()[0]

    def adjust_lr_old(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def train(self, epoch):
        self.model.train()
        # self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            output, target = self.model(data1, data2, data3)
            loss = self.loss(output, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--lr_scheduler', type=str, default='step',
                            help='step or cosine')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001,
                            help='weight decay for optimizer')

        return parser
