from tqdm import tqdm
from src.utils import (
    load_batch_to_device,
    MetricMonitor,
    accuracy,
    plot_classification_results
)
from .base_trainer import BaseTrainer
import torch


class ClassificationTrainer(BaseTrainer):
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
                # compute metrics and loss
                metric_monitor.update("loss", loss.item())
                metrics = metric_monitor.get_metrics()
                metrics["lr"] = self.optimizer.param_groups[0]['lr']
                tepoch.set_postfix(**metrics)

        self.logger.add_metrics(metrics, "train")

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
                    # compute metrics and loss
                    metric_monitor.update("loss", loss.item())
                    metric_monitor.update("acc", accuracy(output, batch["y"]))
                    metrics = metric_monitor.get_metrics()
                    tepoch.set_postfix(**metrics)

        if epoch % self.save_media_epoch == 0:
            classification_results = plot_classification_results(batch["x"], output, batch["y"],
                                                                 self.val_dl.dataset.dataset.labels)
            self.logger.add_media({"classification_results": classification_results})
        self.logger.add_metrics(metrics, "val")

        return metrics["loss"]
