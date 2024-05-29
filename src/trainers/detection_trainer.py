import torch
from matplotlib.figure import Figure
from typing import Dict
import matplotlib.pyplot as plt
import cv2

from .base_trainer import BaseTrainer
from src.metrics import mean_average_precision, MetricMonitor
from src.utils import (
    load_batch_to_device,
    tensors_to_images
)


class DetectionTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            loss_computed_by_model,
            test_dl=None,
            criterion=None,
            scheduler=None
    ):
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.loss_computed_by_model = loss_computed_by_model

    def predict(self, model, batch):
        if self.loss_computed_by_model and model.training:
            return model(batch["x"], batch["y"])
        return model(batch["x"])

    def compute_loss(self, output, batch):
        if self.loss_computed_by_model:
            if isinstance(output, dict):
                return torch.stack([v for _, v in output.items()]).mean()
            return torch.stack(list(output)).sum()
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None if `loss_computed_by_model` is False.")
        return self.criterion(output, batch["y"])

    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        mAP, _ = mean_average_precision(
            [a["boxes"] for a in output],
            [a["labels"] for a in output],
            [a["scores"] for a in output],
            [a["boxes"] for a in batch["y"]],
            [a["labels"] for a in batch["y"]],
            n_classes=self.config.n_classes,
            threshold=0.000001,
        )
        metric_monitor.update("mAP", mAP.item())
        return metric_monitor.get_metrics()

    # def generate_media(self) -> Dict[str, Figure]:
    #     self.model.eval()
    #
    #     # Qualitative results
    #     batch = next(self.val_dl.__iter__())
    #     batch = load_batch_to_device(batch, self.device)
    #     output = self.predict(self.model, batch)
    #
    #     images = tensors_to_images(batch["x"])
    #
    #     detection_results = self.plot_detection_results(images, output, batch["y"], self.config.class_names)
    #
    #     return {"segmentation_results": detection_results}

