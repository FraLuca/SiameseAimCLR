import torch
import multiprocessing as mp
import argparse
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from feeder.ntu_feeder import Feeder_triple
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


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, base_encoder, cfg):
        super().__init__()
        self.model = BYOLAimCLR(base_encoder, **vars(cfg.model_args))
        self.cfg = cfg

        if cfg.resume_from != 'None':
            print("Resume from checkpoint {} in progress...".format(cfg.resume_from))
            state_dict = torch.load(cfg.resume_from)
            self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loading completed!")

        # log hyperparams to wandb
        self.save_hyperparameters()

    def forward(self, batch):
        [data1, data2, data3], label = batch

        data1 = data1.float()
        data2 = data2.float()
        data3 = data3.float()
        label = label.long()

        if self.cfg.stream == 'joint':
            pass
        elif self.cfg.stream == 'motion':
            motion1 = torch.zeros_like(data1)
            motion2 = torch.zeros_like(data2)
            motion3 = torch.zeros_like(data3)

            motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
            motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
            motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]

            data1 = motion1
            data2 = motion2
            data3 = motion3
        elif self.cfg.stream == 'bone':
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

        if self.current_epoch <= self.cfg.mining_epoch:
            loss1, loss2, loss3 = self.model(data1, data2, data3, nnm=False)
            loss = loss1.mean() + (loss2.mean() + loss3.mean()) / 2.  # the mean does the average per each gpu

            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(self.cfg.batch_size)
            else:
                self.model.update_ptr(self.cfg.batch_size)
        else:
            loss1, loss2, loss3, loss4, loss5, loss6 = self.model(
                data1, data2, data3, nnm=True, topk=self.cfg.topk)
            loss_a = loss1.mean() + (loss2.mean() + loss3.mean()) / 2.  # the mean does the average per each gpu
            loss_b = loss4.mean() + (loss5.mean() + loss6.mean()) / 2.  # the mean does the average per each gpu
            loss = (1 - self.cfg.lambda_mining) * loss_a + (self.cfg.lambda_mining) * loss_b

            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(self.cfg.batch_size)
            else:
                self.model.update_ptr(self.cfg.batch_size)

        return loss

    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log('loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.base_lr, momentum=0.9,
                                    nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.step, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


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
                        default='/data_volume/SiameseAimCLR/config/ntu60/pretext/prova.yaml')
    arg = parser.parse_args()

    cfg = load_config(arg)

    # data loading
    train_feeder = Feeder_triple(cfg.train_feeder_args.data_path,
                                 cfg.train_feeder_args.label_path)
    train_loader = DataLoader(
        dataset=train_feeder,
        batch_size=cfg.batch_size // len(cfg.device),
        shuffle=True,
        pin_memory=True,    # set True when memory is abundant
        num_workers=mp.cpu_count()//3,
        persistent_workers=True,
        drop_last=True)

    wandb_logger = None
    if not cfg.disable_wandb:
        # init wandb logger
        wandb_logger = WandbLogger(project='aimclr', group='dev')

    # init self-supervised learner
    model = SelfSupervisedLearner(STGCN, cfg)

    if wandb_logger is not None:
        wandb_logger.watch(model, log_freq=10)

    checkpoint_callback = PeriodicCheckpoint(dirpath=cfg.work_dir, every=cfg.save_interval)

    # init trainer
    trainer = pl.Trainer(
        gpus=cfg.device,
        max_epochs=cfg.num_epoch,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy='ddp',
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])

    # start training
    trainer.fit(model, train_loader)
