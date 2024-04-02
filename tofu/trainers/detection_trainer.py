from tqdm import tqdm
import torch
from .base_trainer import BaseTrainer


class DetectionTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_epoch(self, epoch):
        self.model.train()
        # total_metrics = init_sem_segmentation_metrics()
        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for step, (images, targets) in enumerate(tepoch):
                # Transfer data to the device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                # Clear the gradients
                self.optimizer.zero_grad()
                # Compute the model's predictions and the loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # Backpropagate the gradients
                losses.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # Print loss after every 100 iterations
                if step % 100 == 0:
                    print(f"Epoch {epoch}, iteration {step}, loss = {losses.item()}")
        # if self.log:
        #     self.logger.add_segmentation_table(og_imgs, outputs, targets, "train")
        #     self.logger.add(metrics, "train")
        return losses.item()

    def val_epoch(self, epoch):
        self.model.eval()
        # total_metrics = init_sem_segmentation_metrics()
        with torch.no_grad():
            # use tqdm to track progress
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                # Iterate over data.
                for step, (images, targets) in enumerate(tepoch):
                    # Transfer data to the device
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    # Clear the gradients
                    self.optimizer.zero_grad()
                    # Compute the model's predictions and the loss
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    # Backpropagate the gradients
                    losses.backward()
                    self.optimizer.step()
                    # Print loss after every 100 iterations
                    if step % 100 == 0:
                        print(f"Epoch {epoch}, iteration {step}, loss = {losses.item()}")

        # if self.log:
        #     self.logger.add_segmentation_table(og_imgs, outputs, targets, "val")
        #     self.logger.add(metrics, "val")
        return losses.item()
