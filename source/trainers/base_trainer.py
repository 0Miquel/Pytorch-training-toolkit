from source.utils import *
import math
import os
import time
from omegaconf import OmegaConf
from abc import ABC, abstractmethod


def save_model(model, model_path):
    for path_dir in model_path.split("/")[:-1]:
        os.makedirs(path_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)


class BaseTrainer(ABC):
    def __init__(self, config, wandb_name):
        self.log = False
        if wandb_name is not None:
            # if wandb project name is set or if config is none which means that we are executing a sweep
            self.log = True
            self.logger = get_logger(config, wandb_name)
            self.config = self.logger.cfg
        else:
            self.config = OmegaConf.to_object(config)

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
                save_model(self.model, self.model_path)
            if self.log:
                self.logger.upload()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if self.log:
            self.logger.log_model(self.model_path)
            self.logger.finish()
