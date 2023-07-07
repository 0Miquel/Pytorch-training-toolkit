from tofu.utils import *
from tofu.datasets import get_dataloaders
from tofu.losses import get_loss
from tofu.models import get_model
from tofu.optimizers import get_optimizer
from tofu.schedulers import get_scheduler

import math
import os
import time
from abc import ABC, abstractmethod
import torch


class BaseTrainer(ABC):
    def __init__(self, config):
        set_random_seed(42)

        self.config = config
        trainer_config = config["trainer"]

        self.logger = None
        if trainer_config["wandb"] is not None:
            self.logger = get_logger(config)

        self.n_epochs = trainer_config["n_epochs"]
        self.device = trainer_config["device"]
        self.model_path = trainer_config["model_path"]
        # DATASET
        dataloaders = get_dataloaders(config['dataset'], config["transforms"])
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]
        # LOSS
        self.loss = get_loss(config['loss'])
        # MODEL
        model = get_model(config['model']).to(self.device)
        self.model = torch.compile(model)
        # OPTIMIZER
        self.optimizer = get_optimizer(config['optimizer'], self.model)
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, len(self.train_dl),
                                       n_epochs=self.n_epochs) if "scheduler" in config.keys() else None

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def val_epoch(self, epoch):
        pass

    def fit(self):
        since = time.time()
        best_loss = math.inf
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(self.model, self.model_path)
            if self.logger is not None:
                self.logger.upload()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if self.logger is not None:
            self.logger.log_model(self.model_path)
            self.logger.finish()

    @staticmethod
    def save_model(model, model_path):
        for path_dir in model_path.split("/")[:-1]:
            os.makedirs(path_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
