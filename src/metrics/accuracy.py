import torch


def accuracy(outputs, targets):
    targets_label = torch.max(targets, dim=1)[1]
    outputs_label = torch.max(outputs, dim=1)[1]
    corrects = torch.sum(outputs_label == targets_label).item()
    return corrects / len(targets_label)
