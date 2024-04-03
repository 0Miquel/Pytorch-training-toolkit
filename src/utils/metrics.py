import torch
import torch.nn as nn
import numpy as np


# CLASSIFICATION #######################################################################################################
def init_classification_metrics():
    metrics = {"acc": []}
    return metrics


def compute_classification_metrics(outputs, targets, metrics):
    epoch_metrics = {}
    # ACCURACY
    metrics["acc"].append(accuracy(outputs, targets))
    epoch_metrics["acc"] = np.mean(metrics["acc"])

    return epoch_metrics


# METRIC LEARNING ######################################################################################################
def init_metric_learning_metrics():
    return {}


def compute_metric_learning_metrics(loss, metrics):
    return {}


# SEMANTIC SEGMENTATION ################################################################################################
def init_sem_segmentation_metrics():
    metrics = {"dice": [], "iou": []}
    return metrics


def compute_sem_segmentation_metrics(outputs, targets, metrics):
    epoch_metrics = {}
    # DICE
    metrics["dice"].append(dice_coef(targets, outputs).cpu().detach().numpy())
    epoch_metrics["dice"] = np.mean(metrics["dice"])
    # IOU
    metrics["iou"].append(iou_coef(targets, outputs).cpu().detach().numpy())
    epoch_metrics["iou"] = np.mean(metrics["iou"])

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
