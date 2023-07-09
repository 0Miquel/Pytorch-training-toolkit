import torch
import torch.nn as nn
import numpy as np


# CLASSIFICATION #######################################################################################################
def init_classification_metrics():
    metrics = {"loss": 0, "acc": 0}
    return metrics


def compute_classification_metrics(outputs, targets, total_metrics, step):
    epoch_metrics = {}
    # ACCURACY
    total_metrics["acc"] += accuracy(outputs, targets)
    epoch_metrics["acc"] = total_metrics["acc"] / step

    return epoch_metrics


# METRIC LEARNING ######################################################################################################
def init_metric_learning_metrics():
    metrics = {"loss": 0}
    return metrics


def compute_metric_learning_metrics(loss, total_metrics, step, optimizer=None):
    epoch_metrics = {}
    # LOSS
    total_metrics["loss"] += loss.item()
    epoch_metrics["loss"] = total_metrics["loss"] / step
    # LEARNING RATE
    if optimizer is not None:
        epoch_metrics["lr"] = optimizer.param_groups[0]['lr']

    return epoch_metrics


# SEMANTIC SEGMENTATION ################################################################################################
def init_sem_segmentation_metrics():
    metrics = {"loss": 0, "dice": 0, "iou": 0}
    return metrics


def compute_sem_segmentation_metrics(loss, outputs, targets, total_metrics, step, optimizer=None):
    epoch_metrics = {}
    # LOSS
    total_metrics["loss"] += loss.item()
    epoch_metrics["loss"] = total_metrics["loss"] / step
    # LEARNING RATE
    if optimizer is not None:
        epoch_metrics["lr"] = optimizer.param_groups[0]['lr']
    # DICE
    total_metrics["dice"] += dice_coef(targets, outputs).cpu().detach().numpy()
    epoch_metrics["dice"] = total_metrics["dice"] / step
    # IOU
    total_metrics["iou"] += iou_coef(targets, outputs).cpu().detach().numpy()
    epoch_metrics["iou"] = total_metrics["iou"] / step

    return epoch_metrics

########################################################################################################################


def accuracy(outputs, targets):
    targets_label = torch.max(targets, dim=1)[1]
    outputs_label = torch.max(outputs, dim=1)[1]
    corrects = torch.sum(outputs_label == targets_label).item()
    return corrects / len(targets_label)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = nn.Sigmoid()(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = nn.Sigmoid()(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou
