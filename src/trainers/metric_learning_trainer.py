from tqdm import tqdm
from src.utils import (
    load_batch_to_device,
    MetricMonitor
)
from .base_trainer import BaseTrainer
import torch
from src.miners import get_miner


class MetricLearningTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.miner = get_miner(config["miner"])

    def train_epoch(self, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                output = self.model(batch["x"])
                miner_output = self.miner(output, batch["label"].squeeze())
                # loss
                loss = self.loss(output, batch["label"].squeeze(), miner_output)
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
            self.logger.add(metrics, "train")

        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()

        with torch.no_grad():
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                for step, batch in enumerate(tepoch):
                    batch = load_batch_to_device(batch, self.device)
                    # forward
                    output = self.model(batch["x"])
                    miner_output = self.miner(output, batch["label"].squeeze())
                    # loss
                    loss = self.loss(output, batch["label"].squeeze(), miner_output)
                    # compute epoch loss
                    metric_monitor.update("loss", loss.item())
                    metrics = metric_monitor.get_metrics()
                    tepoch.set_postfix(**metrics)

        if self.logger is not None:
            self.logger.add(metrics, "val")

        return metrics["loss"]
