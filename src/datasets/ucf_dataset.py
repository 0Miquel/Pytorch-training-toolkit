import torch
import cv2
import torch
import numpy as np
import os
import csv
from torch.utils.data import Dataset


class UCFDataset(Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, train, transforms, data_path, n_frames=10):
        if train:
            self.data_path = os.path.join(data_path, "train")
            csv_filename = os.path.join(data_path, "train.csv")
        else:
            self.data_path = os.path.join(data_path, "test")
            csv_filename = os.path.join(data_path, "test.csv")

        with open(csv_filename, newline='') as csvfile:
            # Create a CSV reader object
            csvreader = csv.reader(csvfile)
            # Initialize an empty list to store the values
            labels = []
            video_filenames = []
            for row in csvreader:
                video_filenames.append(row[0])
                labels.append(row[1])

        self.labels = labels[1:]
        video_filenames = video_filenames[1:]
        self.video_filenames = [os.path.join(self.data_path, video_filename)
                                for video_filename in video_filenames]

        self.class_names = np.unique(self.labels).tolist()
        self.n_classes = len(self.class_names)
        self.n_frames = n_frames

        self.transforms = transforms

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        # process image
        video_path = self.video_filenames[idx]
        video = read_video(video_path, self.transforms, self.n_frames)
        if len(video) == 1:
            video = video[0]
        else:
            video = torch.stack(video)
        # process label
        label = self.labels[idx]
        label_idx = self.class_names.index(label)
        encoded_label = np.zeros(self.n_classes, dtype='float32')
        encoded_label[label_idx] = 1

        numeric_label = np.where(encoded_label == 1)[0]

        return {
            "x": video,
            "y": torch.tensor(encoded_label),
            "label": torch.tensor(numeric_label),
        }


def read_video(filename, transforms=None, n_frames=None):
    # Open the video file
    cap = cv2.VideoCapture(filename)

    frames = []

    # Read until video is completed
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame[:, :, ::-1]  # Convert color format from BGR to RGB
        if transforms is not None:
            frame = transforms(image=frame)["image"]
        # Convert frame to numpy array
        frames.append(frame)

    # Release the video capture object
    cap.release()

    idx = np.round(np.linspace(0, len(frames) - 1, n_frames)).astype(int)
    selected_frames = [frames[i] for i in idx]

    return selected_frames
