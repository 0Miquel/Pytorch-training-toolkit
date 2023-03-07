from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
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

resize_transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2(),
])


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

        self.id2labels = {i: label for i, label in enumerate(settings["labels"])}
        self.labels2id = dict((v, k) for k, v in self.id2labels.items())

        self.num_classes = len(self.labels2id)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # process image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        transformed_img = self.transforms(image=img)["image"]
        # process label
        label = self.labels[idx]
        label_id = self.labels2id[label]
        encoded_id = np.zeros(self.num_classes, dtype='float32')
        encoded_id[label_id] = 1

        og_img = resize_transform(image=img)["image"]
        return transformed_img, torch.tensor(encoded_id), og_img