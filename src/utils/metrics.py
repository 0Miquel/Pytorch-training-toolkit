import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class MetricMonitor:
    def __init__(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_metrics(self):
        return {metric_name: metric["avg"] for metric_name, metric in self.metrics.items()}


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
