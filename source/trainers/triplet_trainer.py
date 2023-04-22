from tqdm import tqdm
from source.datasets import get_dataloaders
from source.losses import get_loss
from source.models import get_model
from source.optimizers import get_optimizer
from source.schedulers import get_scheduler
from source.utils import *
from .base_trainer import BaseTrainer


class TripletTrainer(BaseTrainer):
    def __init__(self, config, wandb_name):
        super().__init__(config, wandb_name)
        config = self.config

        trainer_config = config["trainer"]
        self.n_epochs = trainer_config["n_epochs"]
        self.device = trainer_config["device"]
        self.model_path = trainer_config["model_path"]

        dataloaders = get_dataloaders(config['dataset'], config["transforms"])
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]
        self.loss = get_loss(config['loss'])
        model = get_model(config['model'])
        self.model = model.to(self.device)

        self.optimizer = get_optimizer(config['optimizer'], self.model)
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, len(self.train_dl),
                                       n_epochs=self.n_epochs) if "scheduler" in config.keys() else None

    def train_epoch(self, epoch):
        self.model.train()
        total_metrics = init_metric_learning_metrics()
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            for step, (im0, im1, im2) in enumerate(tepoch):
                im0 = im0.to(self.device)
                im1 = im1.to(self.device)
                im2 = im2.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                feat_im0, feat_im1, feat_im2 = self.model(im0, im1, im2)
                # loss
                loss = self.loss(feat_im0, feat_im1, feat_im2)
                # backward
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # compute metrics for this epoch +  current lr and loss
                metrics = compute_metric_learning_metrics(loss, total_metrics, step + 1, self.optimizer)
                tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add(metrics, "train")

        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        total_metrics = init_metric_learning_metrics()
        with torch.no_grad():
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                for step, (im0, im1, im2) in enumerate(tepoch):
                    im0 = im0.to(self.device)
                    im1 = im1.to(self.device)
                    im2 = im2.to(self.device)
                    # forward
                    feat_im0, feat_im1, feat_im2 = self.model(im0, im1, im2)
                    # loss
                    loss = self.loss(feat_im0, feat_im1, feat_im2)
                    # compute metrics for this epoch and loss
                    metrics = compute_metric_learning_metrics(loss, total_metrics, step + 1)
                    tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add(metrics, "val")

        return metrics["loss"]
