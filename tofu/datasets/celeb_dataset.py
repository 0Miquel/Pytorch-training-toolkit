import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class GenerativeCelebDataset(Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, data_path, settings, transforms):
        self.data_path = data_path
        self.transforms = transforms
        self.settings = settings

        self.img_paths = glob.glob(self.data_path + "/*")
        self.img_paths = [path.replace("\\", "/") for path in self.img_paths]  # for Windows

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # process image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        transformed_img = self.transforms(image=img)["image"]

        return {
            "imgs": transformed_img
        }


