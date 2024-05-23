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
import numpy as np
import seaborn as sns
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM


class ClassificationTrainer(BaseTrainer):
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
        if hasattr(self.model, 'target_layers'):
            # initialize a Classification Activation Map if the model has target layers defined
            self.cam = GradCAM(model=self.model, target_layers=self.model.target_layers)
        else:
            self.cam = None

    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> dict:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_pred, y_true, _ = self.post_process(output, batch["y"])

        acc = metrics.accuracy_score(y_true, y_pred)

        metric_monitor.update("acc", acc)
        return metric_monitor.get_metrics()

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
        y_pred, y_true, y_prob = self.post_process(output, batch["y"])
        images = tensors_to_images(batch['x'])
        grayscale_cam = self.cam(input_tensor=batch['x']) if self.cam is not None else None
        classification_results = self.plot_classification_results(images, y_pred, y_true, y_prob,
                                                                  self.val_dl.dataset.labels, grayscale_cam)

        return {"classification_results": classification_results, "confusion_matrix": confusion_matrix}

    @staticmethod
    def post_process(y_pred, y_true):
        y_prob = nn.Softmax(dim=1)(y_pred)
        y_pred = torch.argmax(y_prob, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        return y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), y_prob.detach().cpu().numpy()

    @staticmethod
    def plot_classification_results(images, y_pred, y_true, y_prob, labels, grayscale_cams=None):
        # initialize figure
        ncols = 3 if grayscale_cams is not None else 2
        fig, ax = plt.subplots(nrows=y_pred.shape[0], ncols=ncols, figsize=(ncols*2, y_pred.shape[0]))
        fig.tight_layout()

        # plot figure
        for i, (img, y_pred_, y_true_, y_prob_, cam) in enumerate(zip(images, y_pred, y_true, y_prob)):
            output_label = labels[int(y_pred_)]
            target_label = labels[int(y_true_)]

            max_idx = int(y_pred_)
            bar_colors = ['g' if j == max_idx and output_label == target_label
                          else 'r' if j == max_idx and output_label != target_label else 'b' for j in
                          range(len(labels))]
            if i == 0:
                ax[i, 0].set_title("Image")
                ax[i, 1].set_title("Probabilities")
                if grayscale_cams is not None:
                    ax[i, 2].set_title("CAM")
            ax[i, 0].imshow(img)
            ax[i, 0].axis('off')
            ax[i, 0].set_title(f"{target_label}")
            ax[i, 1].bar(labels, y_prob_, color=bar_colors)
            ax[i, 1].set_ylim(0, 1.0)
            ax[i, 1].tick_params(axis='x', rotation=30)

            if grayscale_cams is not None:
                cam = grayscale_cams[i]
                float_img = img.astype(np.float32) / 255
                cam_visualization = show_cam_on_image(float_img, cam, use_rgb=True)
                ax[i, 2].imshow(cam_visualization)
                ax[i, 2].axis('off')

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
