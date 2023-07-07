from torch.utils.data import DataLoader, Subset
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import importlib
import pkgutil

# get a list of all submodules of the current package
package_name = __name__
package_path = __path__
module_names = [name for _, name, _ in pkgutil.walk_packages(package_path)]


def get_dataloaders(config_dataset, config_transforms):
    dataset_name = config_dataset["dataset_name"]
    settings = config_dataset["settings"]
    train_path = config_dataset["train_path"]
    val_path = config_dataset["val_path"]
    transforms = get_transforms(config_transforms)

    train_dataset = None
    val_dataset = None
    # get loss defined in this package
    for module_name in module_names:
        module = importlib.import_module(f'{package_name}.{module_name}')
        if hasattr(module, dataset_name):
            train_dataset = getattr(module, dataset_name)(train_path, settings, transforms["train"])
            val_dataset = getattr(module, dataset_name)(val_path, settings, transforms["val"])
            break

    if train_dataset is None or val_dataset is None:
        raise f"Dataset with name {dataset_name} not found"
    else:
        # pre-shuffle validation set, it won't be shuffled in eval phase
        val_indices = list(range(len(val_dataset)))
        np.random.shuffle(val_indices)  # Create a fixed permutation of validation indices
        val_dataset = Subset(val_dataset, val_indices)

        train_dl = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=settings["batch_size"], shuffle=False)

        return {"train": train_dl, "val": val_dl}


def get_transforms(config_transforms):
    train_transforms = get_albumentation_transforms(config_transforms["train"])
    val_transforms = get_albumentation_transforms(config_transforms["val"])
    transforms = {"train": train_transforms, "val": val_transforms}
    return transforms


def get_albumentation_transforms(transforms_dict):
    transform = A.Compose([getattr(A, key)(**value) for key, value in transforms_dict.items()] + [ToTensorV2()])
    return transform
