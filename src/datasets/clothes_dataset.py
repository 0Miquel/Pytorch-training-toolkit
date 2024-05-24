from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import os
import torch


class ClothesSegmentationDataset(Dataset):
    def __init__(self, train, transforms, data_path, labels):
        if train:
            self.data_path = os.path.join(data_path, "train")
        else:
            self.data_path = os.path.join(data_path, "val")
        self.labels = labels

        images_dir = os.path.join(self.data_path, "images")
        masks_dir = os.path.join(self.data_path, "masks")
        img_paths = glob.glob(images_dir + "/*.jpeg")
        mask_paths = glob.glob(masks_dir + "/*.png")

        self.img_paths = sorted(img_paths)
        self.mask_paths = sorted(mask_paths)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = mask.astype("float32")

        transformed = self.transforms(image=img, mask=mask)
        transformed_img = transformed["image"]
        mask = transformed["mask"]

        return {
            "x": transformed_img,
            "y": mask
        }
