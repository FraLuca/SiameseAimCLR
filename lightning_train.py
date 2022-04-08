import torch
import multiprocessing as mp
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

from feeder.ntu_feeder import Feeder_triple
from feeder.tools import process_stream
from net.byol_aimclr_lightning import BYOLAimCLR
from net.st_gcn_no_proj import Model as STGCN
from tools import load_config, PeriodicCheckpoint
from net.utils.tools import weights_init


pl.seed_everything(123)
torch.use_deterministic_algorithms(True)


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, base_encoder, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = BYOLAimCLR(base_encoder, **vars(cfg.model_args))
        # self.model.apply(weights_init)

        # self.lambda_nnm = self.cfg.lambda_mining

        if cfg.resume_from != 'None':
            print("Resume from checkpoint {} in progress...".format(cfg.resume_from))
            state_dict = torch.load(cfg.resume_from)
            self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loading completed!")

        # log hyperparams to wandb
        self.save_hyperparameters()

    def forward(self, batch):
        [data1, data2, data3], label = batch

        data1 = process_stream(data1.float(), self.cfg.stream)
        data2 = process_stream(data2.float(), self.cfg.stream)
        data3 = process_stream(data3.float(), self.cfg.stream)
        label = label.long()

        if not self.cfg.model_args.use_nnm:
            loss1, loss2, loss3 = self.model(data1, data2, data3, nnm=False)
            loss = loss1.mean() + (loss2.mean() + loss3.mean()) / 2.

        elif self.current_epoch < self.cfg.mining_epoch:
            loss1, loss2, loss3 = self.model(data1, data2, data3, nnm=False)
            loss = loss1.mean() + (loss2.mean() + loss3.mean()) / 2.

        else:
            loss1, loss2, loss3, loss4, loss5, loss6 = self.model(
                data1, data2, data3, nnm=True, topk=self.cfg.topk)
            loss_a = loss1.mean() + (loss2.mean() + loss3.mean()) / 2.
            loss_b = loss4.mean() + (loss5.mean() + loss6.mean()) / 2.

            self.lambda_nnm = (self.current_epoch - self.cfg.mining_epoch) / \
                (self.cfg.num_epoch - self.cfg.mining_epoch)
            loss = (1 - self.lambda_nnm) * loss_a + (self.lambda_nnm) * loss_b

        return loss

    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log('loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        if self.cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.base_lr, momentum=0.9,
                                        nesterov=self.cfg.nesterov, weight_decay=float(self.cfg.weight_decay))
        else:
            raise ValueError("Invalid optimizer {}".format(self.cfg.optimizer))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.step, gamma=0.1)

        return [optimizer], [lr_scheduler]

    def on_before_zero_grad(self, _):
        if self.model.use_momentum:
            self.model.update_moving_average()


if __name__ == '__main__':

    cfg = load_config(name='BYOL AimCLR Training')

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
    learner = SelfSupervisedLearner(STGCN, cfg)

    if wandb_logger is not None:
        wandb_logger.watch(learner, log_freq=10)

    checkpoint_callback = PeriodicCheckpoint(dirpath=cfg.work_dir, every=cfg.save_interval)

    # init trainer
    trainer = pl.Trainer(
        gpus=cfg.device,
        max_epochs=cfg.num_epoch,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy=DDPPlugin(find_unused_parameters=False),
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])

    # start training
    trainer.fit(learner, train_loader)
