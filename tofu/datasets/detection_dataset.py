from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DetectionDataset(Dataset):
    def __init__(self, path, settings, transforms):
        self.path = path
        self.transforms = transforms
        self.settings = settings

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
