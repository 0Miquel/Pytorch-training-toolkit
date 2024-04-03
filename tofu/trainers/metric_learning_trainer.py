from tqdm import tqdm
from tofu.utils import load_batch_to_device, init_metric_learning_metrics, compute_metric_learning_metrics
from .base_trainer import BaseTrainer
import torch
from tofu.miners import get_miner


class MetricLearningTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.miner = get_miner(config["miner"])

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
                    output = self.model(batch["x"])
                    miner_output = self.miner(output, batch["label"].squeeze())
                    # loss
                    loss = self.loss(output, batch["label"].squeeze(), miner_output)
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
