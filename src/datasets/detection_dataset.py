from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
import os
from xml.etree import ElementTree as et
import albumentations as A
from albumentations.pytorch import ToTensorV2


# the dataset class
class DetectionDataset(Dataset):
    def __init__(self, train, data_path, classes, batch_size, transforms=None):
        if train:
            self.dir_path = data_path + "/train"
            self.transforms = A.Compose([
                A.Resize(width=224, height=224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))
        else:
            self.dir_path = data_path + "/val"
            self.transforms = A.Compose([
                A.Resize(width=224, height=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))

        self.batch_size = batch_size
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        # read the image
        image = cv2.imread(image_path)[:, :, ::-1]  # convert it to rgb

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        # apply the image transforms
        sample = self.transforms(image=image,
                                 bboxes=boxes,
                                 labels=labels)
        input_image = sample['image']
        target = {
            'boxes': torch.as_tensor(sample['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(sample['labels'], dtype=torch.int64)
        }

        return target

    def __len__(self):
        return len(self.all_images)
