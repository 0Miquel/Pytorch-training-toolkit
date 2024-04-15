from src.utils import (
    MetricMonitor,
    dice_coef,
    iou_coef,
    plot_segmentation_results,
    load_batch_to_device
)
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict


class SegmentationTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, sample) -> None:
        """
        Update metric_monitor with the metrics computed from output and sample.
        """
        metric_monitor.update("dice", dice_coef(sample["y"], output).item())
        metric_monitor.update("iou", iou_coef(sample["y"], output).item())

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and sample.
        """
        self.model.eval()
        sample = next(self.val_dl.__iter__())
        sample = load_batch_to_device(sample, self.device)
        output = self.predict(self.model, sample)
        segmentation_results = plot_segmentation_results(sample["x"], output, sample["y"])
        return {"segmentation_results": segmentation_results}
