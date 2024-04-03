import glob
import cv2
import torch
import numpy as np
import os
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, train, transforms, data_path, labels, batch_size):
        if train:
            self.data_path = os.path.join(data_path, "train")
        else:
            self.data_path = os.path.join(data_path, "val")
        self.batch_size = batch_size
        self.labels = labels
        self.num_classes = len(self.labels)

        self.img_paths = glob.glob(self.data_path + "/*/*")
        self.gt = [path.split("/")[-2] for path in self.img_paths]

        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # process image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        transformed_img = self.transforms(image=img)["image"]
        # process label
        label = self.gt[idx]
        label_idx = self.labels.index(label)
        encoded_id = np.zeros(self.num_classes, dtype='float32')
        encoded_id[label_idx] = 1

        return {
            "x": transformed_img,
            "y": torch.tensor(encoded_id)
        }
