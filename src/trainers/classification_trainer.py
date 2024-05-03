from src.utils import (
    load_batch_to_device,
    tensors_to_images
)
from src.metrics import MetricMonitor, accuracy
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


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
        confusion_matrix = self.plot_confusion_matrix(actual, predicted, self.val_dl.dataset.labels)

        # Qualitative results
        sample = next(self.val_dl.__iter__())
        sample = load_batch_to_device(sample, self.device)
        output = self.predict(self.model, sample)
        classification_results = self.plot_classification_results(sample["x"], output, sample["y"],
                                                                  self.val_dl.dataset.labels)

        return {"classification_results": classification_results, "confusion_matrix": confusion_matrix}

    @staticmethod
    def plot_classification_results(x, y_pred, y_true, labels):
        images = tensors_to_images(x)
        y_pred = nn.Softmax(dim=1)(y_pred)

        fig, ax = plt.subplots(nrows=y_pred.shape[0], ncols=2, figsize=(2, y_pred.shape[0]))
        fig.tight_layout()

        for i, (img, y_pred_, y_true_) in enumerate(zip(images, y_pred, y_true)):
            if i == 0:
                ax[i, 0].set_title("Image")
                ax[i, 1].set_title("Class Probabilities")

            output_label = labels[torch.argmax(y_pred_).item()]
            target_label = labels[torch.argmax(y_true_).item()]

            max_idx = torch.argmax(y_pred_)
            bar_colors = ['g' if j == max_idx and output_label == target_label
                          else 'r' if j == max_idx and output_label != target_label else 'b' for j in
                          range(len(labels))]
            ax[i, 0].imshow(img)
            ax[i, 0].axis('off')
            ax[i, 0].set_title(f"{target_label}")
            ax[i, 1].bar(labels, y_pred_.cpu().detach().numpy(), color=bar_colors)
            ax[i, 1].set_ylim(0, 1.0)
            ax[i, 1].tick_params(axis='x', rotation=30)

        plt.close('all')
        return fig

    @staticmethod
    def plot_confusion_matrix(actual, predicted, classes):
        if isinstance(actual, list):
            actual = torch.cat(actual)
        if isinstance(predicted, list):
            predicted = torch.cat(predicted)

        predicted = nn.Softmax(dim=1)(predicted)
        predicted = torch.argmax(predicted, dim=1)
        predicted = predicted.detach().cpu().numpy()
        actual = actual.detach().cpu().numpy()

        confusion_matrix = metrics.confusion_matrix(actual, predicted, normalize='true')
        cm_figure = sns.heatmap(confusion_matrix, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
        cm_figure.set(xlabel="Predicted", ylabel="True")
        cm_figure = cm_figure.get_figure()

        plt.close('all')
        return cm_figure
