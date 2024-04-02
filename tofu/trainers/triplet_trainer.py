from tqdm import tqdm
from tofu.utils import load_batch_to_device, init_metric_learning_metrics, compute_metric_learning_metrics
from .base_trainer import BaseTrainer
import torch


class TripletTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0

        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                feat_im0, feat_im1, feat_im2 = self.model(batch["anchors"], batch["positives"], batch["negatives"])
                # loss
                loss = self.loss(feat_im0, feat_im1, feat_im2)
                # backward
                loss.backward()
                # optimize
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # compute epoch loss
                running_loss += loss.item()
                epoch_loss = running_loss / (step + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                tepoch.set_postfix({"loss": epoch_loss, "lr": current_lr})

        if self.logger is not None:
            self.logger.add({"loss": epoch_loss, "lr": current_lr}, "train")

        return epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        metrics = init_metric_learning_metrics()
        running_loss = 0

        with torch.no_grad():
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                for step, batch in enumerate(tepoch):
                    batch = load_batch_to_device(batch, self.device)
                    # forward
                    feat_im0, feat_im1, feat_im2 = self.model(batch["anchors"], batch["positives"], batch["negatives"])
                    # loss
                    loss = self.loss(feat_im0, feat_im1, feat_im2)
                    # compute epoch loss
                    running_loss += loss.item()
                    epoch_loss = running_loss / (step + 1)
                    # compute metrics for this epoch and loss
                    epoch_metrics = compute_metric_learning_metrics(loss, metrics)
                    epoch_metrics["loss"] = epoch_loss
                    tepoch.set_postfix(**epoch_metrics)

        if self.logger is not None:
            self.logger.add(metrics, "val")

        return epoch_loss
