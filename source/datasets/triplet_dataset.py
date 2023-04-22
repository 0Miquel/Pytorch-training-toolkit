from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


default_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

resize_transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2(),
])


class TripletMITDataset(Dataset):
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        # GET ANCHOR IMG
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        transformed_anchor = self.transforms(image=img)["image"]
        anchor_label = self.labels[idx]
        # GET POSITIVE IMG
        while True:
            positive_idx = random.randint(0, len(self.labels) - 1)
            positive_label = self.labels[positive_idx]
            if positive_idx != idx and positive_label == anchor_label:
                positive_img_path = self.img_paths[positive_idx]
                postiive_img = cv2.imread(positive_img_path)[:, :, ::-1]
                transformed_positive = self.transforms(image=postiive_img)["image"]
                break
        # GET NEGATIVE IMG
        while True:
            negative_idx = random.randint(0, len(self.labels) - 1)
            negative_label = self.labels[negative_idx]
            if negative_label != anchor_label:
                negative_img_path = self.img_paths[negative_idx]
                negative_img = cv2.imread(negative_img_path)[:, :, ::-1]
                transformed_negative = self.transforms(image=negative_img)["image"]
                break

        return transformed_anchor, transformed_positive, transformed_negative
