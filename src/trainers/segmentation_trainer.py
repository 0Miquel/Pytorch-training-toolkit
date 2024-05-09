from src.utils import (
    load_batch_to_device,
    colors,
    tensors_to_images,
)
from src.metrics import MetricMonitor
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class SegmentationMultiTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> None:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_pred, y_true, n_classes = self.post_process(output, batch["y"])
        tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode='multiclass', num_classes=n_classes)

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("f1", f1_score.item())
        metric_monitor.update("iou", iou_score.item())

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        self.model.eval()
        batch = next(self.val_dl.__iter__())
        batch = load_batch_to_device(batch, self.device)
        output = self.predict(self.model, batch)

        y_pred, y_true, n_classes = self.post_process(output, batch["y"])
        segmentation_results = self.plot_segmentation_results(batch["x"], y_pred, y_true, n_classes)

        return {"segmentation_results": segmentation_results}

    @staticmethod
    def post_process(y_pred, y_true):
        n_classes = y_pred.shape[1]
        y_pred = nn.Softmax(dim=1)(y_pred)
        y_pred = torch.argmax(y_pred, dim=1)  # Get the index of the maximum value along the class dimension
        y_true = torch.argmax(y_true, dim=1)
        return y_pred.long(), y_true.long(), n_classes

    @staticmethod
    def plot_segmentation_results(x, y_pred, y_true, n_classes):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        imgs = tensors_to_images(x)

        fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=3, figsize=(3, y_pred.shape[0]))
        fig.tight_layout()

        for i, (img, y_pred_, y_true_) in enumerate(zip(imgs, y_pred, y_true)):
            # Create the mask images
            mask_pred = np.zeros((y_true_.shape[0], y_true_.shape[1], 3), dtype='uint8')
            mask_true = np.zeros((y_true_.shape[0], y_true_.shape[1], 3), dtype='uint8')
            for j in range(n_classes):
                color = colors[j % len(colors)]
                mask_true[y_true_ == j] = color
                mask_pred[y_pred_ == j] = color

            # Show the mask images in the plot
            if i == 0:
                axes[i, 0].set_title("Image")
                axes[i, 1].set_title("Ground truth")
                axes[i, 2].set_title("Predicted")
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask_true)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(mask_pred)
            axes[i, 2].axis('off')

        plt.close('all')
        return fig


class SegmentationBinaryTrainer(BaseTrainer):
    def compute_metrics(self, metric_monitor: MetricMonitor, output, batch) -> None:
        """
        Update metric_monitor with the metrics computed from output and batch.
        """
        y_pred, y_true = self.post_process(output, batch["y"])
        tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode='binary')

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("f1", f1_score.item())
        metric_monitor.update("iou", iou_score.item())

    def generate_media(self) -> Dict[str, Figure]:
        """
        Generate media from output and batch.
        """
        self.model.eval()
        batch = next(self.val_dl.__iter__())
        batch = load_batch_to_device(batch, self.device)
        output = self.predict(self.model, batch)

        y_pred, y_true = self.post_process(output, batch["y"])
        segmentation_results = self.plot_segmentation_results(batch["x"], y_pred, y_true)

        return {"segmentation_results": segmentation_results}

    @staticmethod
    def post_process(y_pred, y_true, thr=0.5):
        y_pred = nn.Sigmoid()(y_pred)
        y_pred = y_pred > thr
        return y_pred.long(), y_true.long()

    @staticmethod
    def plot_segmentation_results(x, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        imgs = tensors_to_images(x)

        fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=3, figsize=(3, y_pred.shape[0]))
        fig.tight_layout()

        for i, (img, y_pred_, y_true_) in enumerate(zip(imgs, y_pred, y_true)):
            # Create the mask images
            mask_pred = np.zeros((y_true_.shape[1], y_true_.shape[2], 3), dtype='uint8')
            mask_true = np.zeros((y_true_.shape[1], y_true_.shape[2], 3), dtype='uint8')
            mask_true[y_true_[0] == 1] = [255, 255, 255]
            mask_pred[y_pred_[0] == 1] = [255, 255, 255]

            # Show the mask images in the plot
            if i == 0:
                axes[i, 0].set_title("Image")
                axes[i, 1].set_title("Ground truth")
                axes[i, 2].set_title("Predicted")
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask_true)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(mask_pred)
            axes[i, 2].axis('off')

        plt.close('all')
        return fig
