import torch
import torch.nn as nn
import numpy as np

EXEC_PARAMS = {}


def compute_metrics(metrics, outputs, targets, inputs, loss, optimizer=None):
    global EXEC_PARAMS

    results = {}

    # compute epoch loss
    EXEC_PARAMS["dataset_size"] += inputs.size(0)
    EXEC_PARAMS["running_loss"] += loss.item() * inputs.size(0)
    epoch_loss = EXEC_PARAMS["running_loss"] / EXEC_PARAMS["dataset_size"]
    results["loss"] = epoch_loss
    # get current learning rate if optimizer (training phase)
    if optimizer is not None:
        current_lr = optimizer.param_groups[0]['lr']
        results["lr"] = current_lr

    if "accuracy" in metrics:
        # Get target and predicted labels for metrics
        targets_label = torch.max(targets, dim=1)[1]
        outputs_label = torch.max(outputs, dim=1)[1]
        # Accuracy
        corrects = torch.sum(outputs_label == targets_label).item()
        EXEC_PARAMS["running_corrects"] += corrects
        epoch_acc = EXEC_PARAMS["running_corrects"] / EXEC_PARAMS["dataset_size"]
        results["acc"] = epoch_acc
    if "dice" in metrics:
        y_pred = nn.Sigmoid()(outputs)
        val_dice = dice_coef(targets, y_pred).cpu().detach().numpy()
        EXEC_PARAMS["total_dice"].append(val_dice)
        epoch_dice = np.mean(EXEC_PARAMS["total_dice"])
        results["dice"] = epoch_dice
    if "iou" in metrics:
        y_pred = nn.Sigmoid()(outputs)
        val_jaccard = iou_coef(targets, y_pred).cpu().detach().numpy()
        EXEC_PARAMS["total_iou"].append(val_jaccard)
        epoch_iou = np.mean(EXEC_PARAMS["total_iou"])
        results["iou"] = epoch_iou
    return results


def init_exec_params(metrics):
    global EXEC_PARAMS
    EXEC_PARAMS = {"running_loss": 0.0, "dataset_size": 0}

    if "accuracy" in metrics:
        EXEC_PARAMS["running_corrects"] = 0
    if "dice" in metrics:
        EXEC_PARAMS["total_dice"] = []
    if "iou" in metrics:
        EXEC_PARAMS["total_iou"] = []


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
