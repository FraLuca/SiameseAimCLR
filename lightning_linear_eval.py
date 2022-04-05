from tabnanny import verbose
import torch
import multiprocessing as mp
import argparse
import yaml
import numpy as np

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
from torch import nn
from collections import OrderedDict

from feeder.ntu_feeder import Feeder_single
from net.byol_aimclr_lightning import BYOLAimCLR
from net.st_gcn_no_proj import Model as STGCN


pl.seed_everything(123)


def load_config(arg):
    # load YAML config file as dict
    with open(arg.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)

    # load new parser with default arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.set_defaults(**default_arg)
    try:
        cfg = parser.parse_args('')
    except Exception as e:
        print(e)

    # build sub-parsers
    for k, value in cfg._get_kwargs():
        if isinstance(value, dict):
            new_parser = argparse.ArgumentParser(add_help=False)
            new_parser.set_defaults(**value)
            cfg.__setattr__(k, new_parser.parse_args(''))

    return cfg


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, base_encoder, cfg):
        super().__init__()
        self.model = BYOLAimCLR(base_encoder, **vars(cfg.model_args))
        self.cfg = cfg
        self.best_top1_acc = 0.

        self.load_weights(cfg.weights, cfg.ignore_weights)

        for name, param in self.model.online_encoder.named_parameters():
            if not str(name).startswith('projector'):
                param.requires_grad = False
        self.num_grad_layers = 2

        if hasattr(self.model, 'encoder_q_motion'):
            for name, param in self.model.encoder_q_motion.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2
        if hasattr(self.model, 'encoder_q_bone'):
            for name, param in self.model.encoder_q_bone.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2

        self.loss = nn.CrossEntropyLoss()
        self.top1_accuracy = Accuracy(top_k=1)
        self.top5_accuracy = Accuracy(top_k=5)
        # print(self.model)

        # log hyperparams to wandb
        self.save_hyperparameters()

    def load_weights(self, weights_path, ignore_weights=None):
        print("Loading weights from {} ...".format(weights_path))
        weights = torch.load(weights_path)['state_dict']
        weights = OrderedDict([[k.split('model.')[-1],
                                v.cpu()] for k, v in weights.items()])

        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                print('Filter [{}] remove weights [{}].'.format(i, n))

        try:
            self.model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                print('Can not find weights [{}].'.format(d))
            state.update(weights)
            self.model.load_state_dict(state)

        print("Weights loading completed!")

    def forward(self, data):
        if self.cfg.stream == 'joint':
            pass
        elif self.cfg.stream == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif self.cfg.stream == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone = torch.zeros_like(data)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

            data = bone
        else:
            raise ValueError

        # forward
        output = self.model(None, data, return_projection=True)
        return output

    def training_step(self, batch, _):
        data, label = batch[0].float(), batch[1].long()
        output = self.forward(data)
        loss = self.loss(output, label)
        self.log('loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        data, label = batch[0].float(), batch[1].long()
        output = self.forward(data)
        loss = self.loss(output, label)
        self.log('val_loss', loss)
        return {'loss': loss, 'pred': output, 'target': label}

    def validation_epoch_end(self, outputs):

        top1_acc = 0.
        top5_acc = 0.
        loss_epoch = 0.

        for output in outputs:
            loss = output['loss']
            acc1 = self.top1_accuracy(output['pred'], output['target'])
            acc5 = self.top5_accuracy(output['pred'], output['target'])
            loss_epoch += loss
            top1_acc += acc1
            top5_acc += acc5

        loss_epoch = round(loss_epoch.item() / len(outputs), 4)
        top1_acc = round((top1_acc.item() / len(outputs)) * 100, 2)
        top5_acc = round((top5_acc.item() / len(outputs)) * 100, 2)

        if top1_acc > self.best_top1_acc:
            self.best_top1_acc = top1_acc

        log_dict = {
            'loss_epoch': loss_epoch,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'best_top1_acc': self.best_top1_acc
        }

        self.log_dict(log_dict, sync_dist=True)
        self.print("\n----- Results epoch {} -----".format(self.current_epoch))
        for k, v in log_dict.items():
            self.print(k, '\t=\t', v)
        self.print()

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.base_lr, momentum=0.9,
                                    nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.step, gamma=0.1, verbose=False)
        return [optimizer], [lr_scheduler]


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str,  every: int):
        super().__init__()
        self.dirpath = dirpath
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"epoch-{pl_module.current_epoch}.ckpt"
            trainer.save_checkpoint(current)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BYOL AimCLR Training')
    parser.add_argument('-c', '--config', type=str, required=True,
                        default='/data_volume/SiameseAimCLR/config/ntu60/linear_eval/prova.yaml')
    arg = parser.parse_args()

    cfg = load_config(arg)

    # train data loading
    train_feeder = Feeder_single(cfg.train_feeder_args.data_path,
                                 cfg.train_feeder_args.label_path,
                                 cfg.train_feeder_args.shear_amplitude,
                                 cfg.train_feeder_args.temperal_padding_ratio)
    train_loader = DataLoader(
        dataset=train_feeder,
        batch_size=cfg.batch_size // len(cfg.device),
        shuffle=True,
        pin_memory=True,    # set True when memory is abundant
        num_workers=mp.cpu_count()//3,
        persistent_workers=True,
        drop_last=True)

    # validation data loading
    test_feeder = Feeder_single(cfg.test_feeder_args.data_path,
                                cfg.test_feeder_args.label_path,
                                cfg.test_feeder_args.shear_amplitude,
                                cfg.test_feeder_args.temperal_padding_ratio)
    test_loader = DataLoader(
        dataset=test_feeder,
        batch_size=cfg.test_batch_size // len(cfg.device),
        shuffle=False,
        pin_memory=True,    # set True when memory is abundant
        num_workers=mp.cpu_count()//3,
        persistent_workers=True,
        drop_last=True)

    wandb_logger = None
    if not cfg.disable_wandb:
        # init wandb logger
        wandb_logger = WandbLogger(project='aimclr', group='eval')

    # init self-supervised learner
    learner = SelfSupervisedLearner(STGCN, cfg)

    if wandb_logger is not None:
        wandb_logger.watch(learner, log_freq=10)

    # checkpoint_callback = PeriodicCheckpoint(dirpath=cfg.work_dir, every=cfg.save_interval)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.work_dir, save_top_k=1,
                                          verbose=True, monitor='top1_acc',
                                          filename='{epoch}-{top1_acc:.2f}')

    # init trainer
    trainer = pl.Trainer(
        gpus=cfg.device,
        max_epochs=cfg.num_epoch,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy=DDPPlugin(find_unused_parameters=False),
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.eval_interval,
    )

    # start training
    trainer.fit(learner, train_loader, test_loader)
    # trainer.save_checkpoint(cfg.work_dir + '/best_model.ckpt')
