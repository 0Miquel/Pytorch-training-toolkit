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
from sklearn import metrics
import seaborn as sns


class ClassificationTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> None:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_pred, y_true, _ = self.post_process(output, batch["y"])

        acc = metrics.accuracy_score(y_true, y_pred)

        metric_monitor.update("acc", acc)

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        self.model.eval()

        # Confusion matrix
        predicted = []
        actual = []
        for step, batch in enumerate(self.val_dl):
            batch = load_batch_to_device(batch, self.device)
            # forward
            output = self.predict(self.model, batch)
            predicted.append(output)
            actual.append(batch["y"])
        actual = torch.cat(actual)
        predicted = torch.cat(predicted)
        predicted, actual, _ = self.post_process(predicted, actual)
        confusion_matrix = self.plot_confusion_matrix(actual, predicted, self.val_dl.dataset.labels)

        # Qualitative results
        batch = next(self.val_dl.__iter__())
        batch = load_batch_to_device(batch, self.device)
        output = self.predict(self.model, batch)
        y_pred, y_true, y_pred_prob = self.post_process(output, batch["y"])
        classification_results = self.plot_classification_results(batch["x"], y_pred, y_true, y_pred_prob,
                                                                  self.val_dl.dataset.labels)

        return {"classification_results": classification_results, "confusion_matrix": confusion_matrix}

    @staticmethod
    def post_process(y_pred, y_true):
        y_pred_prob = nn.Softmax(dim=1)(y_pred)
        y_pred = torch.argmax(y_pred_prob, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        return y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), y_pred_prob.detach().cpu().numpy()

    @staticmethod
    def plot_classification_results(x, y_pred, y_true, probabilities, labels):
        images = tensors_to_images(x)

        fig, ax = plt.subplots(nrows=y_pred.shape[0], ncols=2, figsize=(4, y_pred.shape[0]))
        fig.tight_layout()

        for i, (img, y_pred_, y_true_, y_prob) in enumerate(zip(images, y_pred, y_true, probabilities)):
            output_label = labels[int(y_pred_)]
            target_label = labels[int(y_true_)]

            max_idx = int(y_pred_)
            bar_colors = ['g' if j == max_idx and output_label == target_label
                          else 'r' if j == max_idx and output_label != target_label else 'b' for j in
                          range(len(labels))]
            if i == 0:
                ax[i, 0].set_title("Image")
                ax[i, 1].set_title("Class Probabilities")
            ax[i, 0].imshow(img)
            ax[i, 0].axis('off')
            ax[i, 0].set_title(f"{target_label}")
            ax[i, 1].bar(labels, y_prob, color=bar_colors)
            ax[i, 1].set_ylim(0, 1.0)
            ax[i, 1].tick_params(axis='x', rotation=30)

        plt.close('all')
        return fig

    @staticmethod
    def plot_confusion_matrix(actual, predicted, classes):
        confusion_matrix = metrics.confusion_matrix(actual, predicted, normalize='true')
        cm_figure = sns.heatmap(confusion_matrix, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
        cm_figure.set(xlabel="Predicted", ylabel="True")
        cm_figure = cm_figure.get_figure()

        plt.close('all')
        return cm_figure
