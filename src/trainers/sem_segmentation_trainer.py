from tqdm import tqdm
from src.utils import (
    load_batch_to_device,
    MetricMonitor,
    dice_coef,
    iou_coef
)
from .base_trainer import BaseTrainer
import torch


class SemSegmentationTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_epoch(self, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                output = self.model(batch["x"])
                # loss
                loss = self.loss(output, batch["y"])
                # backward
                loss.backward()
                # optimize
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # compute epoch loss
                metric_monitor.update("loss", loss.item())
                metrics = metric_monitor.get_metrics()
                metrics["lr"] = self.optimizer.param_groups[0]['lr']
                tepoch.set_postfix(**metrics)

        if self.logger is not None:
            self.logger.add_segmentation_table(batch["x"], output, batch["y"], "train")
            self.logger.add(metrics, "train")

        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()

        with torch.no_grad():
            # use tqdm to track progress
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                # Iterate over data.
                for step, batch in enumerate(tepoch):
                    batch = load_batch_to_device(batch, self.device)
                    # predict
                    output = self.model(batch["x"])
                    # loss
                    loss = self.loss(output, batch["y"])
                    # compute epoch loss
                    metric_monitor.update("loss", loss.item())
                    metric_monitor.update("dice", dice_coef(batch["y"], output).item())
                    metric_monitor.update("iou", iou_coef(batch["y"], output).item())
                    metrics = metric_monitor.get_metrics()
                    tepoch.set_postfix(**metrics)

        if self.logger is not None:
            self.logger.add_segmentation_table(batch["x"], output, batch["y"], "val")
            self.logger.add(metrics, "val")

        return metrics["loss"]
