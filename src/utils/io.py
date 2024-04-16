from yaml.loader import SafeLoader
import yaml
import torch
import random
import numpy as np
import argparse
import os
import hydra


def save_figure(figure, figure_name):
    outputs_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    figs_dir = os.path.join(outputs_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    figure_path = os.path.join(figs_dir, figure_name)
    figure.savefig(figure_path)


def load_yaml_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_batch_to_device(batch, device):
    """
    Load batch to device recursively in case it finds a list or dictionary in such batch of data,
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        new_batch = {}
        for key, value in batch.items():
            new_batch[key] = load_batch_to_device(value, device)
        return new_batch
    elif isinstance(batch, list):
        new_batch = []
        for item in batch:
            new_batch.append(load_batch_to_device(item, device))
        return new_batch
    else:
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="config_classification")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--sweep", default="sweep_classification.yaml")
    parser.add_argument("--n_runs", default=5, type=int)

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
