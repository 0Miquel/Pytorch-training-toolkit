import torch
import torch.nn as nn
import numpy as np


def compute_metrics(exec_metrics, metrics, outputs, targets, loss, optimizer=None):
    results = {}

    # compute epoch loss
    exec_metrics["dataset_size"] += outputs.size(0)
    exec_metrics["running_loss"] += loss.item() * outputs.size(0)
    epoch_loss = exec_metrics["running_loss"] / exec_metrics["dataset_size"]
    results["loss"] = epoch_loss
    # get current learning rate if optimizer (training phase)
    if optimizer is not None:
        current_lr = optimizer.param_groups[0]['lr']
        results["lr"] = current_lr

    if "accuracy" in metrics:
        # Get target and predicted labels for metrics
        targets_label = torch.max(targets, dim=1)[1]
        outputs_label = torch.max(outputs, dim=1)[1]
        exec_metrics["predictions"] = exec_metrics["predictions"] + outputs_label.detach().cpu().tolist()
        exec_metrics["gt"] = exec_metrics["gt"] + targets_label.detach().cpu().tolist()
        # Accuracy
        corrects = torch.sum(outputs_label == targets_label).item()
        exec_metrics["running_corrects"] += corrects
        epoch_acc = exec_metrics["running_corrects"] / exec_metrics["dataset_size"]
        results["acc"] = epoch_acc
    if "dice" in metrics:
        y_pred = nn.Sigmoid()(outputs)
        val_dice = dice_coef(targets, y_pred).cpu().detach().numpy()
        exec_metrics["total_dice"].append(val_dice)
        epoch_dice = np.mean(exec_metrics["total_dice"])
        results["dice"] = epoch_dice
    if "iou" in metrics:
        y_pred = nn.Sigmoid()(outputs)
        val_jaccard = iou_coef(targets, y_pred).cpu().detach().numpy()
        exec_metrics["total_iou"].append(val_jaccard)
        epoch_iou = np.mean(exec_metrics["total_iou"])
        results["iou"] = epoch_iou
    return results, exec_metrics


def init_exec_params(metrics):
    exec_metrics = {"running_loss": 0.0, "dataset_size": 0}

    if "accuracy" in metrics:
        exec_metrics["running_corrects"] = 0
        exec_metrics["predictions"] = []
        exec_metrics["gt"] = []
    if "dice" in metrics:
        exec_metrics["total_dice"] = []
    if "iou" in metrics:
        exec_metrics["total_iou"] = []

    return exec_metrics


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou
