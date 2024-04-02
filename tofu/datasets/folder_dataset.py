import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, train, transforms, train_path, val_path, labels, batch_size):
        if train:
            self.data_path = train_path
        else:
            self.data_path = val_path
        self.transforms = transforms
        self.batch_size = batch_size
        self.labels = labels

        self.img_paths = glob.glob(self.data_path + "/*/*")
        self.img_paths = [path.replace("\\", "/") for path in self.img_paths]  # for Windows
        self.gt = [path.split("/")[-2] for path in self.img_paths]

        self.num_classes = len(self.labels)

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
            "imgs": transformed_img,
            "labels": torch.tensor(encoded_id)
        }
