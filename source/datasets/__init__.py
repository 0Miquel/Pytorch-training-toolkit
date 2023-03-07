from .folder_dataset import FolderDataset
from .flood_dataset import FloodAreaSegmentation

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import torch
import numpy as np


# set random seed for numpy
np.random.seed(42)

# set random seed for PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_dataloaders(config):
    dataset_name = config["dataset_name"]
    settings = config["settings"]
    validation = config["validation"]

    try:
        if validation:
            train_dl, val_dl = get_predefined_split(dataset_name, settings)
        else:
            train_dl, val_dl = get_random_split(dataset_name, settings)
    except KeyError:
        raise f"Dataset with name {dataset_name} not found"

    return {"train": train_dl, "val": val_dl}


def get_predefined_split(dataset_name, settings):
    train_path = settings["train_path"]
    val_path = settings["val_path"]
    train_dataset = globals()[dataset_name](train_path, settings)
    val_dataset = globals()[dataset_name](val_path, settings)
    # pre-shuffle validation set, it won't be shuffled in training/eval phase
    val_indices = list(range(len(val_dataset)))
    np.random.shuffle(val_indices)  # Create a fixed permutation of validation indices
    val_dataset = Subset(val_dataset, val_indices)

    train_dl = DataLoader(train_dataset, batch_size=settings["train_batch_size"], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=settings["val_batch_size"], shuffle=False)

    return train_dl, val_dl


def get_random_split(dataset_name, settings):
    train_ratio = settings["train_ratio"]
    dataset = globals()[dataset_name](settings)
    train_size = int(train_ratio * len(dataset))
    # shuffle indices
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    train_dataset = Subset(dataset, train_indices)
    # generate reproducible validation set
    val_indices = indices[train_size:]
    val_dataset = Subset(dataset, val_indices)

    train_dl = DataLoader(train_dataset, batch_size=settings["train_batch_size"], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=settings["val_batch_size"], shuffle=False)

    return train_dl, val_dl