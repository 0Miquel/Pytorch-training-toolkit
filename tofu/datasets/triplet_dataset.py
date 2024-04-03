from torch.utils.data import Dataset
import glob
import cv2
import random


class TripletMITDataset(Dataset):
    def __init__(self, data_path, settings, transforms):
        self.data_path = data_path
        self.transforms = transforms
        self.settings = settings

        self.img_paths = glob.glob(self.data_path + "/*/*")
        self.labels = [path.split("/")[-2] for path in self.img_paths]

        self.id2labels = {i: label for i, label in enumerate(self.settings["labels"])}
        self.labels2id = dict((v, k) for k, v in self.id2labels.items())

        self.num_classes = len(self.labels2id)

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
                positive_img = cv2.imread(positive_img_path)[:, :, ::-1]
                transformed_positive = self.transforms(image=positive_img)["image"]
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

        return {
            "anchors": transformed_anchor,
            "positives": transformed_positive,
            "negatives": transformed_negative
        }
