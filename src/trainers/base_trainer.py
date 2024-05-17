from tqdm import tqdm
import torch
from typing import Dict
from matplotlib.figure import Figure

from src.metrics import MetricMonitor
from src.utils import (
    load_batch_to_device,
    Logger,
    EarlyStopping,
    ModelCheckpoint,
    set_random_seed
)


class BaseTrainer:
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            criterion=None,
            scheduler=None,
            test_dl=None,
    ):
        set_random_seed(42)

        self.config = config
        self.n_epochs = config.n_epochs
        self.device = config.device
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            monitor=config.monitor,
            max_mode=config.max_mode
        )
        self.model_checkpoint = ModelCheckpoint(monitor=config.monitor, max_mode=config.max_mode)
        self.logger = Logger(config)

        # DATASET
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # LOSS FUNCTION
        self.criterion = criterion

        # MODEL
        self.model = model.to(self.device)
        # self.model = torch.compile(self.model)

        # OPTIMIZER
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                output = self.predict(self.model, batch)
                # loss
                loss = self.compute_loss(output, batch)
                # backward
                loss.backward()
                # optimize
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # update loss and learning rate
                metric_monitor.update("loss", loss.item())
                metrics = metric_monitor.get_metrics()
                metrics["lr"] = self.optimizer.param_groups[0]['lr']
                tepoch.set_postfix(**metrics)

        return metrics

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        metric_monitor = MetricMonitor()

        # use tqdm to track progress
        with tqdm(self.val_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
            # Iterate over data.
            for step, batch in enumerate(tepoch):
                batch = load_batch_to_device(batch, self.device)
                # predict
                output = self.predict(self.model, batch)
                # loss
                if self.criterion is not None:
                    loss = self.compute_loss(output, batch)
                    # update metrics and loss
                    metric_monitor.update("loss", loss.item())
                metrics = self.compute_metrics(metric_monitor, output, batch)
                tepoch.set_postfix(**metrics)

        return metrics

    def fit(self):
        for epoch in range(self.n_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)

            # upload metrics to wandb and save locally
            self.logger.upload_metrics(train_metrics, val_metrics, epoch)

            # save model and early stop callbacks
            self.model_checkpoint(self.model, val_metrics)
            early_stop = self.early_stopping(epoch, val_metrics)
            if early_stop:
                break

        if self.test_dl is not None:
            self.evaluate()

        self.logger.finish()

        return self.model_checkpoint.best_metric

    def predict(self, model, batch):
        return model(batch["x"])

    def compute_loss(self, output, batch):
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None.")
        return self.criterion(output, batch["y"])

    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch and
        return the metrics as a dictionary.
        """
        return {}

    def evaluate(self):
        print("Loading best model and generating media...")
        self.model_checkpoint.load_best_model(self.model)
        figures = self.generate_media()
        self.logger.upload_media(figures)

    @torch.no_grad()
    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        return {}
