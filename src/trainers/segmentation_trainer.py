from src.utils import (
    load_batch_to_device,
    colors,
    tensors_to_images,
)
from src.metrics import MetricMonitor, dice_coef, iou_coef
from .base_trainer import BaseTrainer
from matplotlib.figure import Figure
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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

        binary = sample["y"].shape[1] == 1
        y_true, y_pred = self.post_process_results(output, sample["y"], binary)
        labels = self.val_dl.dataset.labels

        segmentation_results = self.plot_segmentation_results(sample["x"], y_pred, y_true, labels)
        return {"segmentation_results": segmentation_results}

    @staticmethod
    def post_process_results(y_pred, y_true, binary):
        if binary:
            # binary segmentation
            y_pred = nn.Sigmoid()(y_pred)
            y_pred = (y_pred > 0.5).squeeze()
            y_pred = y_pred.detach().cpu().numpy().astype("float32")

            y_true = y_true.squeeze()
            y_true = y_true.detach().cpu().numpy().astype("float32")
        else:
            # multi-class segmentation
            y_pred = nn.Softmax(dim=1)(y_pred)
            y_pred = torch.argmax(y_pred, axis=1)
            y_pred = y_pred.detach().cpu().numpy().astype("float32")

            y_true = torch.argmax(y_true, axis=1)
            y_true = y_true.detach().cpu().numpy().astype("float32")

        return y_true, y_pred

    @staticmethod
    def plot_segmentation_results(x, y_pred, y_true, labels):
        imgs = tensors_to_images(x)

        fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=3, figsize=(3, y_pred.shape[0]))
        fig.tight_layout()

        for i, (img, y_pred_, y_true_) in enumerate(zip(imgs, y_pred, y_true)):
            # Create the mask images
            mask_pred = np.zeros((y_true_.shape[0], y_true_.shape[1], 3), dtype='uint8')
            mask_true = np.zeros((y_true_.shape[0], y_true_.shape[1], 3), dtype='uint8')
            for j, label in enumerate(labels):
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
