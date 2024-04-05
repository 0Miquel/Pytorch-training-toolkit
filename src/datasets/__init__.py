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


def get_dataloaders(cfg_dataset, cfg_transforms):
    dataset_name = cfg_dataset["dataset_name"]
    settings = cfg_dataset["settings"] if "settings" in cfg_dataset.keys() else {}
    transforms = get_transforms(cfg_transforms)

    train_dataset = None
    val_dataset = None

    # get dataset defined in this package
    for module_name in module_names:
        module = importlib.import_module(f'{package_name}.{module_name}')
        if hasattr(module, dataset_name):
            train_dataset = getattr(module, dataset_name)(train=True, transforms=transforms["train"], **settings)
            val_dataset = getattr(module, dataset_name)(train=False, transforms=transforms["val"], **settings)
            break

    # build dataloaders
    if train_dataset is None or val_dataset is None:
        raise AttributeError(f"Dataset with name {dataset_name} not found")
    else:
        train_dl = DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True, drop_last=True)
        # pre-shuffle validation set, it won't be shuffled in eval phase
        val_indices = list(range(len(val_dataset)))
        np.random.shuffle(val_indices)  # Create a fixed permutation of validation indices
        val_dataset = Subset(val_dataset, val_indices).dataset
        val_dl = DataLoader(val_dataset, batch_size=val_dataset.batch_size, shuffle=False, drop_last=True)

        return {"train": train_dl, "val": val_dl}


def get_transforms(cfg_transforms):
    transforms = {}

    if "train" in cfg_transforms.keys():
        train_transforms = get_albumentations_transforms(cfg_transforms["train"])
        transforms["train"] = train_transforms
    else:
        transforms["train"] = A.Compose([ToTensorV2()])

    if "val" in cfg_transforms.keys():
        val_transforms = get_albumentations_transforms(cfg_transforms["val"])
        transforms["val"] = val_transforms
    else:
        transforms["val"] = A.Compose([ToTensorV2()])

    return transforms


def get_albumentations_transforms(transforms_dict):
    transform = A.Compose([getattr(A, key)(**value) for key, value in transforms_dict.items()] + [ToTensorV2()])
    return transform
