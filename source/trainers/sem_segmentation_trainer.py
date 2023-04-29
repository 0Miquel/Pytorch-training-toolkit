from tqdm import tqdm
from source.utils import *
from .base_trainer import BaseTrainer


class SemSegmentationTrainer(BaseTrainer):
    def __init__(self, config, wandb_name):
        super().__init__(config, wandb_name)

    def train_epoch(self, epoch):
        self.model.train()
        total_metrics = init_sem_segmentation_metrics()
        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for step, (inputs, targets, og_imgs) in enumerate(tepoch):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                outputs = self.model(inputs)
                # loss
                loss = self.loss(outputs, targets)
                # backward
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # compute metrics for this epoch +  current lr and loss
                metrics = compute_sem_segmentation_metrics(loss, outputs, targets, total_metrics, step+1, self.optimizer)
                tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add_segmentation_table(og_imgs, outputs, targets, "train")
            self.logger.add(metrics, "train")
        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        total_metrics = init_sem_segmentation_metrics()
        with torch.no_grad():
            # use tqdm to track progress
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                # Iterate over data.
                for step, (inputs, targets, og_imgs) in enumerate(tepoch):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # predict
                    outputs = self.model(inputs)
                    # loss
                    loss = self.loss(outputs, targets)
                    # compute metrics for this epoch and loss
                    metrics = compute_sem_segmentation_metrics(loss, outputs, targets, total_metrics, step + 1)
                    tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add_segmentation_table(og_imgs, outputs, targets, "val")
            self.logger.add(metrics, "val")
        return metrics["loss"]
