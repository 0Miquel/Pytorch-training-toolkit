import torch

from .base_trainer import BaseTrainer
from src.metrics import mean_average_precision, MetricMonitor


class DetectionTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            train_dl,
            val_dl,
            model,
            optimizer,
            loss_computed_by_model,
            n_classes,
            criterion=None,
            scheduler=None
    ):
        super().__init__(
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            criterion=criterion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.n_classes = n_classes
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

    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> None:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        mAP, _ = mean_average_precision(
            [a["boxes"] for a in output],
            [a["labels"] for a in output],
            [a["scores"] for a in output],
            [a["boxes"] for a in batch["y"]],
            [a["labels"] for a in batch["y"]],
            n_classes=self.n_classes,
            threshold=0.000001,
        )
        metric_monitor.update("mAP", mAP.item())
