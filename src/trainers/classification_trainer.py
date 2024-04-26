from src.utils import (
    load_batch_to_device,
    plot_classification_results,
    plot_confusion_matrix
)
from src.metrics import MetricMonitor, accuracy
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict


class ClassificationTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, sample) -> None:
        """
        Update metric_monitor with the metrics computed from output and sample.
        """
        metric_monitor.update("acc", accuracy(output, sample["y"]))

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and sample.
        """
        self.model.eval()

        # Confussion matrix
        predicted = []
        actual = []
        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            # forward
            output = self.predict(self.model, batch)
            predicted.append(output)
            actual.append(batch["label"].squeeze())
        confusion_matrix = plot_confusion_matrix(actual, predicted, self.val_dl.dataset.labels)

        # Qualitative results
        sample = next(self.val_dl.__iter__())
        sample = load_batch_to_device(sample, self.device)
        output = self.predict(self.model, sample)
        classification_results = plot_classification_results(sample["x"], output, sample["y"],
                                                             self.val_dl.dataset.labels)

        return {"classification_results": classification_results, "confusion_matrix": confusion_matrix}
