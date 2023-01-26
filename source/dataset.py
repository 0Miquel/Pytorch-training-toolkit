from torch.utils.data import Dataset, DataLoader
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

    try:
        train_dataset = globals()[dataset_name](settings["train_path"])
        val_dataset = globals()[dataset_name](settings["val_path"])
    except KeyError:
        raise f"Dataset with name {dataset_name} not found"

    train_dl = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=settings["batch_size"], shuffle=True)
    return {"train": train_dl, "val": val_dl}


class FolderDataset(Dataset):
    def __init__(self, path, transforms=default_transform):
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
