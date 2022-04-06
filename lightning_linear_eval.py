import torch
import multiprocessing as mp
import argparse

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torch import nn
from collections import OrderedDict

from feeder.ntu_feeder import Feeder_single
from feeder.tools import process_stream
from net.byol_aimclr_lightning import BYOLAimCLR
from net.st_gcn_no_proj import Model as STGCN
from tools import load_config
from net.utils.tools import weights_init


pl.seed_everything(123)


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, base_encoder, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = BYOLAimCLR(base_encoder, **vars(cfg.model_args))

        self.load_weights(cfg.weights, cfg.ignore_weights)

        self.loss = nn.CrossEntropyLoss()
        self.top1_accuracy = Accuracy(top_k=1)
        self.top5_accuracy = Accuracy(top_k=5)
        self.best_top1_acc = 0.

        self.save_hyperparameters()

    def load_weights(self, weights_path, ignore_weights=None):
        print("Loading weights from {} ...".format(weights_path))
        weights = torch.load(weights_path, map_location='cpu')['state_dict']
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

        self.model = self.model.cuda()
        # self.model.apply(weights_init)

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

    def forward(self, data):
        data = process_stream(data, self.cfg.stream)
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


if __name__ == '__main__':

    cfg = load_config(name='BYOL AimCLR Linear Evaluation')

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

    checkpoint_callback = ModelCheckpoint(dirpath=cfg.work_dir, save_top_k=1,
                                          verbose=True, monitor='top1_acc', mode='max',
                                          filename='{epoch}-{top1_acc:.2f}')
    # lr_monitor = LearningRateMonitor(logging_interval='step')

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
