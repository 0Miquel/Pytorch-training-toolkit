from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import glob
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


default_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def get_dataloaders(config):
    dataset_name = config["dataset_name"]
    settings = config["settings"]
    validation = config["validation"]

    try:
        if validation:
            train_path = settings["train_path"]
            val_path = settings["val_path"]
            train_dataset = globals()[dataset_name](train_path, settings)
            val_dataset = globals()[dataset_name](val_path, settings)

            train_dl = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
            val_dl = DataLoader(val_dataset, batch_size=settings["batch_size"], shuffle=True)
        else:
            train_ratio = settings["train_ratio"]
            dataset = globals()[dataset_name](settings)
            train_size = int(train_ratio * len(dataset))
            train_sampler = SubsetRandomSampler(range(train_size))
            val_sampler = SubsetRandomSampler(range(train_size, len(dataset)))

            train_dl = DataLoader(dataset, batch_size=settings["batch_size"], sampler=train_sampler)
            val_dl = DataLoader(dataset, batch_size=settings["batch_size"], sampler=val_sampler)
    except KeyError:
        raise f"Dataset with name {dataset_name} not found"

    return {"train": train_dl, "val": val_dl}


class FolderDataset(Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, path, settings, transforms=default_transform):
        if path[-1] != "/":
            path = path + "/"
        self.img_paths = glob.glob(path + "*/*")
        self.img_paths = [path.replace("\\", "/") for path in self.img_paths]  # for Windows
        self.labels = [path.split("/")[-2] for path in self.img_paths]

        self.labelstoid = {path.split("/")[-2]: 0 for path in self.img_paths}
        for i, j in enumerate(self.labelstoid.keys()):
            self.labelstoid[j] = i
        self.idtolabels = dict((v, k) for k, v in self.labelstoid.items())

        self.num_classes = len(self.labelstoid)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # process image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        img = self.transforms(image=img)["image"]
        # process label
        label = self.labels[idx]
        label_id = self.labelstoid[label]
        encoded_id = np.zeros(self.num_classes, dtype='float32')
        encoded_id[label_id] = 1
        return img, torch.tensor(encoded_id)


class FolderDataset2(Dataset):
    """
    FolderDataset but without predefined validation set
    """
    def __init__(self, settings, transforms=default_transform):
        path = settings["path"]
        if path[-1] != "/":
            path = path + "/"
        self.img_paths = glob.glob(path + "*/*")
        self.img_paths = [path.replace("\\", "/") for path in self.img_paths]  # for Windows
        self.labels = [path.split("/")[-2] for path in self.img_paths]

        self.labelstoid = {path.split("/")[-2]: 0 for path in self.img_paths}
        for i, j in enumerate(self.labelstoid.keys()):
            self.labelstoid[j] = i
        self.idtolabels = dict((v, k) for k, v in self.labelstoid.items())

        self.num_classes = len(self.labelstoid)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # process image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        img = self.transforms(image=img)["image"]
        # process label
        label = self.labels[idx]
        label_id = self.labelstoid[label]
        encoded_id = np.zeros(self.num_classes, dtype='float32')
        encoded_id[label_id] = 1
        return img, torch.tensor(encoded_id)


class FloodAreaSegmentation(Dataset):
    def __init__(self, settings, transforms=default_transform):
        images_path = settings["images_path"]
        masks_path = settings["masks_path"]
        if images_path[-1] != "/":
            images_path = images_path + "/"
        if masks_path[-1] != "/":
            masks_path = masks_path + "/"

        img_paths = glob.glob(images_path + "*.jpg")
        mask_paths = glob.glob(masks_path + "*.png")

        img_paths = [path.replace("\\", "/") for path in img_paths]  # for Windows
        mask_paths = [path.replace("\\", "/") for path in mask_paths]  # for Windows

        self.img_paths = sorted(img_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.mask_paths = sorted(mask_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))

        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask.astype('float32')

        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        if len(mask.shape) == 2:
            mask = mask[None, ...]

        return img, mask
