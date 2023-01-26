import torch


def compute_metrics(metrics, outputs, targets, acc):
    results = {}

    if "accuracy" in metrics:
        # Get target and predicted labels for metrics
        targets_label = torch.max(targets, dim=1)[1]
        outputs_label = torch.max(outputs, dim=1)[1]
        # Accuracy
        corrects = torch.sum(outputs_label == targets_label).item()
        acc["running_corrects"] += corrects
        epoch_acc = acc["running_corrects"] / acc["dataset_size"]
        results["acc"] = epoch_acc
    else:
        pass

    return results


def init_exec_params(metrics):
    exec_params = {"running_loss": 0.0, "dataset_size": 0}

    if "accuracy" in metrics:
        exec_params["running_corrects"] = 0
    else:
        pass

    return exec_params
