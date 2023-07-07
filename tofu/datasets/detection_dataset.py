from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
import random
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


class DetectionDataset(Dataset):
    def __init__(self, path, settings, transforms=default_transform):
        self.path = path
        self.transforms = transforms
        self.settings = settings

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
