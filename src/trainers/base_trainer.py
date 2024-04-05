from src.utils import Logger, save_model, set_random_seed
from src.datasets import get_dataloaders
from src.losses import get_loss
from src.models import get_model
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

import time
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config):
        set_random_seed(42)

        self.config = config
        trainer_config = config["trainer"]
        self.n_epochs = trainer_config["n_epochs"]
        self.save_media_epoch = self.n_epochs // 10 if self.n_epochs // 10 > 0 else 1
        self.device = trainer_config["device"]

        self.logger = Logger(config)

        # DATASET
        dataloaders = get_dataloaders(config['dataset'], config["transforms"])
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]

        # LOSS
        self.loss = get_loss(config['loss'])

        # MODEL
        self.model = get_model(config['model']).to(self.device)
        # self.model = torch.compile(self.model)

        # OPTIMIZER
        self.optimizer = get_optimizer(config['optimizer'], self.model)
        total_steps = len(self.train_dl) * self.n_epochs
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, total_steps) \
            if "scheduler" in config.keys() else None

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def val_epoch(self, epoch):
        pass

    def fit(self):
        since = time.time()
        best_loss = 0

        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(self.model, 'best_epoch.pt')
            save_model(self.model, 'last_epoch.pt')

            if self.logger is not None:
                self.logger.upload(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.logger is not None:
            self.logger.finish()

        return best_loss
