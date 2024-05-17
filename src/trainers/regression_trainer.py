from src.utils import (
    load_batch_to_device,
    tensors_to_images
)
from src.metrics import MetricMonitor
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import numpy as np


class RegressionTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_true = batch["y"].detach().cpu().numpy()
        y_pred = output.detach().cpu().numpy()
        class_names = self.val_dl.dataset.class_names
        for i, class_name in enumerate(class_names):
            metric_monitor.update(f"{class_name}_mape",
                                  sklearn.metrics.mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]))

        return metric_monitor.get_metrics()

    @torch.no_grad()
    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        self.model.eval()

        y_true = []
        y_pred = []
        datetimes = []
        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            output = self.predict(self.model, batch)

            y_true.append(batch["y"].detach().cpu().numpy())
            y_pred.append(output.detach().cpu().numpy())
            datetimes.append(batch["datetime"])

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        datetimes = np.hstack(datetimes)

        # TODO: Generate media given the predictions, ground truth and datetimes

        return {}

