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
        transformed_img = transformed["image"]
        mask = transformed["mask"]
        if len(mask.shape) == 2:
            mask = mask[None, ...]

        og_img = resize_transform(image=img)["image"]

        return transformed_img, mask, og_img
