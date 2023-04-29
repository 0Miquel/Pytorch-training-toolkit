from tqdm import tqdm
from source.utils import *
from .base_trainer import BaseTrainer


class TripletTrainer(BaseTrainer):
    def __init__(self, config, wandb_name):
        super().__init__(config, wandb_name)

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
