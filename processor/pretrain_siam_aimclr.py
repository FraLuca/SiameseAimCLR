import argparse
import numpy as np
import wandb

# torch
import torch

# torchlight
from torchlight import str2bool

from .processor import Processor
from .pretrain import PT_Processor


class SiameseAimCLR_Processor(PT_Processor):
    """
        Processor for SiameseAimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)
                motion3 = torch.zeros_like(data3)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
                data3 = motion3
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)
                bone3 = torch.zeros_like(data3)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :,
                                                      v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :,
                                                      v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                    bone3[:, :, :, v1 - 1, :] = data3[:, :, :,
                                                      v1 - 1, :] - data3[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
                data3 = bone3
            else:
                raise ValueError

            # forward
            q, q_extreme, q_extreme_drop, k = self.model(data1, data2, data3)

            # TOMORROW BREAKPOINT LUCA
            # from sklearn.manifold import TSNE
            # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
            # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

            loss_1 = self.sim_loss(q, k).mean()
            loss_2 = self.sim_loss(q_extreme, k).mean()
            loss_3 = self.sim_loss(q_extreme_drop, k).mean()
            loss = (loss_1 + loss_2 + loss_3) / 3.

            # breakpoint in case of loss NaN
            if loss.item() != loss.item():
                # for name, param in self.model.module.encoder_q.named_parameters():
                #     print(name, "\n", param)
                
                for name, param in self.model.module.encoder_q.named_parameters():
                    print(name, "-->", any(param.flatten() > 10**12))

                breakpoint()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            # log on wandb
            if not self.disable_wandb:
                wandb.log(dict(loss=loss.item()))

        # self.adjust_lr()
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
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')

        return parser
