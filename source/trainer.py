from tqdm import tqdm
from source.dataset import *
from source.loss import *
from source.model import *
from source.optimizer import *
from source.scheduler import *
from source.logger import *
from source.utils.metrics import *
from source.utils.plots import *
import math
import time
from omegaconf import OmegaConf


def train(wandb_name=None, cfg=None):
    trainer = Trainer(cfg, wandb_name)
    trainer.fit()


class Trainer:
    def __init__(self, config, wandb_name):
        self.log = False
        if wandb_name is not None:
            # if wandb project name is set or if config is none which means that we are executing a sweep
            self.log = True
            self.logger = get_logger(config, wandb_name)
            config = self.logger.cfg
        else:
            config = OmegaConf.to_object(config)

        trainer_config = config["trainer"]
        self.metrics = trainer_config["metrics"]
        self.n_epochs = trainer_config["n_epochs"]
        self.device = trainer_config["device"]
        self.model_path = trainer_config["model_path"]

        dataloaders = get_dataloaders(config['dataset'])
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]
        self.loss = get_loss(config['loss'])
        model = get_model(config['model'])
        self.model = model.to(self.device)
        self.optimizer = get_optimizer(config['optimizer'], self.model)
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, len(self.train_dl),
                                       n_epochs=self.n_epochs) if "scheduler" in config.keys() else None

    def train_epoch(self, epoch):
        self.model.train()
        init_exec_params(self.metrics)
        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for inputs, targets in tepoch:
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
                metrics = compute_metrics(self.metrics, outputs, targets, inputs, loss, self.optimizer)
                tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add_img("Segmentation train result", plot_segmentation_batch(outputs, targets))
            self.logger.add({"train": metrics})
        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        init_exec_params(self.metrics)
        # use tqdm to track progress
        with tqdm(self.val_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
            # Iterate over data.
            for inputs, targets in tepoch:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # predict
                outputs = self.model(inputs)
                # loss
                loss = self.loss(outputs, targets)

                # compute metrics for this epoch +  current lr and loss
                metrics = compute_metrics(self.metrics, outputs, targets, inputs, loss)
                tepoch.set_postfix(**metrics)
        if self.log:
            self.logger.add_img("Segmentation val result", plot_segmentation_batch(outputs, targets))
            self.logger.add({"val": metrics})
        return metrics["loss"]

    def fit(self):
        since = time.time()
        best_loss = math.inf
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
            if self.log:
                self.logger.upload()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if self.log:
            self.logger.log_model(self.model_path)
            self.logger.finish()
