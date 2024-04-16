from .base_trainer import BaseTrainer
from src.utils import (
    MetricMonitor,
)


class DetectionTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        trainer_config = config["trainer"]
        self.loss_computed_by_model = trainer_config["loss_computed_by_model"] \
            if "loss_computed_by_model" in trainer_config.keys() else False

    def predict(self, model, sample):
        if self.loss_computed_by_model and model.training:
            return model(sample["x"], sample["y"])
        return model(sample["x"])

    def compute_loss(self, output, sample):
        if self.loss_computed_by_model:
            if isinstance(output, dict):
                return None
            return None
        return self.loss(output, sample["y"])

    def compute_metrics(self, metric_monitor: MetricMonitor, output, sample) -> dict:
        """
        Update metric_monitor with the metrics computed from output and sample.
        """
        # TODO: implement detection metrics
        return {}
