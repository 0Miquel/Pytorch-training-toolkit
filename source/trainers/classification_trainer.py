from tqdm import tqdm
from source.datasets import get_dataloaders
from source.losses import get_loss
from source.models import get_model
from source.optimizers import get_optimizer
from source.schedulers import get_scheduler
from source.utils import *
from .base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
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
        # summary(self.model, input_size=(3, 224, 224), device=self.device)
        self.optimizer = get_optimizer(config['optimizer'], self.model)
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, len(self.train_dl),
                                       n_epochs=self.n_epochs) if "scheduler" in config.keys() else None

    def train_epoch(self, epoch):
        self.model.train()
        total_metrics = init_classification_metrics()
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
                metrics = compute_classification_metrics(loss, outputs, targets, total_metrics, step+1, self.optimizer)
                tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add(metrics, "train")
        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        total_metrics = init_classification_metrics()
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
                    metrics = compute_classification_metrics(loss, outputs, targets, total_metrics, step + 1)
                    tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add(metrics, "val")
        return metrics["loss"]
