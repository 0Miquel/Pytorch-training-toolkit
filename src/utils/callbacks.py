from math import inf
from pathlib import Path
import os

import hydra
import torch


class EarlyStopping:
    def __init__(
        self,
        monitor: str,
        max_mode: bool = False,
        min_delta: float = 0.0,
        patience: int = 8,
    ) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.max_mode = max_mode
        self.monitor = monitor

        self.counter: int = 0
        self.best_score: float = -inf if max_mode else inf
        self.early_stop: bool = False

    def __call__(self, epoch: int, metric: dict):
        score = metric[self.monitor]
        is_best = score > self.best_score if self.max_mode else score < self.best_score
        if is_best:
            self.best_score = score
            self.counter = 0
        else:
            is_within_delta = (
                score > (self.best_score - self.min_delta)
                if self.max_mode
                else score < (self.best_score + self.min_delta)
            )
            if not is_within_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        if self.early_stop:
            print(f"Early-Stopping the training process. Epoch: {epoch}.")
            return True

        return False


class ModelCheckpoint:
    def __init__(
            self,
            monitor: str,
            max_mode: bool = False,
    ) -> None:
        self.filepath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/ckpts'
        # track best model
        self.monitor = monitor
        self.max_mode = max_mode
        self.best_metric: float = -inf if max_mode else inf

    def __call__(self, model, metrics: dict) -> None:
        Path(self.filepath).mkdir(parents=True, exist_ok=True)
        score = metrics[self.monitor]
        # save the last model
        filename = os.path.join(self.filepath, 'last.pt')
        torch.save(model.state_dict(), filename)
        # save if the model is the best
        is_best = score > self.best_metric if self.max_mode else score < self.best_metric
        if is_best:
            self.best_metric = score
            filename = os.path.join(self.filepath, 'best.pt')
            torch.save(model.state_dict(), filename)

    def load_best_model(self, model):
        filename = os.path.join(self.filepath, 'best.pt')
        model.load_state_dict(torch.load(filename))

    @staticmethod
    def load_from_pretrained(model, path):
        model.load_state_dict(torch.load(path))
