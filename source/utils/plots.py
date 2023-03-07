import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from abc import ABC
import sklearn
import cv2


def plot_segmentation_batch(y_pred, y_true, thr=0.5):
    y_pred = nn.Sigmoid()(y_pred)
    fig, axes = plt.subplots(nrows=y_pred.shape[0], ncols=2, figsize=(20, 20))
    for i, (y_pred_, y_true_) in enumerate(zip(y_pred, y_true)):
        y_pred_ = y_pred_.permute((1, 2, 0)).cpu().detach().numpy()
        y_true_ = y_true_.permute((1, 2, 0)).cpu().detach().numpy()
        y_pred_ = (y_pred_ > thr).astype(np.float32)
        if i == 0:
            axes[i, 0].set_title("Ground truth")
            axes[i, 1].set_title("Predicted")
        axes[i, 0].imshow(y_true_)
        axes[i, 1].imshow(y_pred_)
        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
    plt.close('all')
    return fig


def segmentation_table(inputs, outputs, targets, labels):
    """
    Creates WandB table
    """
    table = wandb.Table(columns=["Prediction", "Ground truth"])

    for img, pred_mask, true_mask in zip(inputs, outputs, targets):
        pred_mask = nn.Sigmoid()(pred_mask)
        pred_mask = pred_mask.permute((1, 2, 0)).cpu().detach().numpy()
        true_mask = true_mask.permute((1, 2, 0)).cpu().detach().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.float32)
        true_mask = np.squeeze(true_mask).astype("uint8")
        pred_mask = np.squeeze(pred_mask).astype("uint8")
        img = img.permute((1, 2, 0)).cpu().detach().numpy().astype("uint8")

        pred_mask_img = wandb.Image(img, masks={"predictions": {"mask_data": pred_mask, "class_labels": labels}})
        true_mask_img = wandb.Image(img, masks={"ground_truth": {"mask_data": true_mask, "class_labels": labels}})

        table.add_data(pred_mask_img, true_mask_img)

    return table


def classificiation_table(inputs, outputs, targets, labels):
    table = wandb.Table(columns=["Input", "Prediction", "Ground truth", "Probabilities"])
    outputs = nn.Softmax(dim=1)(outputs)
    for img, output, target in zip(inputs, outputs, targets):
        img = img.permute((1, 2, 0)).cpu().detach().numpy().astype("uint8")
        img = wandb.Image(img)
        output_label = labels[torch.argmax(output).item()]
        target_label = labels[torch.argmax(target).item()]

        # create the bar chart
        fig, ax = plt.subplots(figsize=(10, 10))
        max_idx = torch.argmax(output)
        bar_colors = ['g' if i == max_idx and output_label == target_label
                      else 'r' if i == max_idx and output_label != target_label else 'b' for i in range(len(labels))]
        ax.bar([*labels.values()], output.cpu().detach().numpy(), color=bar_colors)
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        ax.tick_params(axis='x', rotation=90)

        # save the chart as a WandB plot
        probabilities = wandb.Image(fig)
        plt.close(fig)
        table.add_data(img, output_label, target_label, probabilities)

    return table



