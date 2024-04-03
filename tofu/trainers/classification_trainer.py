from tqdm import tqdm
from tofu.utils import load_batch_to_device, init_classification_metrics, compute_classification_metrics
from .base_trainer import BaseTrainer
import torch


class ClassificationTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0

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
                running_loss += loss.item()
                epoch_loss = running_loss / (step+1)
                current_lr = self.optimizer.param_groups[0]['lr']
                tepoch.set_postfix({"loss": epoch_loss, "lr": current_lr})

        if self.logger is not None:
            self.logger.add_classification_table(batch["x"], output, batch["y"], "train")
            self.logger.add({"loss": epoch_loss, "lr": current_lr}, "train")

        return epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        metrics = init_classification_metrics()
        running_loss = 0

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
                    running_loss += loss.item()
                    epoch_loss = running_loss / (step + 1)
                    # compute metrics for this epoch and loss
                    epoch_metrics = compute_classification_metrics(output, batch["y"], metrics)
                    epoch_metrics["loss"] = epoch_loss
                    tepoch.set_postfix(**epoch_metrics)

        if self.logger is not None:
            self.logger.add_classification_table(batch["x"], output, batch["y"], "val")
            self.logger.add(epoch_metrics, "val")

        return epoch_loss
