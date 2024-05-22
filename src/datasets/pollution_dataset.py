import glob
import cv2
import torch
import numpy as np
from pathlib import Path
from netCDF4 import Dataset
import os
from datetime import datetime


class PollutionDataset(torch.utils.data.Dataset):
    """
    Class for the typical Folder Dataset, where a folder consists of multiple subfolders for every class which
    contains the class images. It does not support any other format like csv file.
    """
    def __init__(self, train, transforms, data_path, class_names, cams, level):
        if train:
            # self.data_path = os.path.join(data_path, "train")
            ads_path = os.path.join(data_path, "ADS-Dataset")
            images_path = os.path.join(data_path, "train")
        else:
            # self.data_path = os.path.join(data_path, "val")
            ads_path = os.path.join(data_path, "ADS-Dataset")
            images_path = os.path.join(data_path, "val")

        ads_filenames = [os.path.join(ads_path, ads_filename) for ads_filename in os.listdir(ads_path)]
        self.ads_data = {Path(ads_filename).stem: ads_filename for ads_filename in ads_filenames}

        img_paths = []
        for cam in cams:
            img_paths += glob.glob(f"{images_path}/*/{cam}/*")

        datetime_strings = []
        for img_path in img_paths:
            split_img_path = img_path.split("/")
            date = split_img_path[-3]
            filename = split_img_path[-1]
            split_filename = filename.split(" ")

            time = Path(split_filename[2]).stem
            h, m, _ = time.split("-")
            h, m = int(h), float(m)
            m = int(round(m / 10) * 10)
            if m == 60:
                h += 1
                m = 0

            datetime_strings.append(f"{date}_{h}-{m}")

        date_format = "%Y-%m-%d_%H-%M"
        indexed_datetimes = [(index, datetime.strptime(date_str, date_format)) for index, date_str in
                             enumerate(datetime_strings)]
        sorted_indexed_datetimes = sorted(indexed_datetimes, key=lambda x: x[1])
        sorted_indexes = [index for index, _ in sorted_indexed_datetimes]
        sorted_datetimes = [date for _, date in sorted_indexed_datetimes]
        sorted_datetime_strings = [datetime.strftime(date_obj, date_format) for date_obj in sorted_datetimes]
        sorted_img_paths = [img_paths[index] for index in sorted_indexes]

        self.img_path = sorted_img_paths
        self.datetime = sorted_datetime_strings
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.cams = cams
        self.level = level
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        # Get atmospheric data given the datetime of the image
        datetime_str = self.datetime[idx]
        date = datetime_str.split("_")[0]
        time = datetime_str.split("_")[1]
        h, m = time.split("-")
        h, m = float(h), float(m)

        ads_data = self.ads_data[date]
        output_gt = []
        with Dataset(ads_data, mode="r") as data:
            level_idx = np.where(self.level == data.variables['level'][:].data)[0][0]
            for pollutants in self.class_names:
                if pollutants in data.variables:
                    values = data.variables[pollutants][:].data[int(h), level_idx]
                    next_values = data.variables[pollutants][:].data[int(h + 1), level_idx]
                    values = np.mean(values)
                    next_values = np.mean(next_values)
                    interpolated_values = self.interpolate_values(values, next_values)
                    final_value = interpolated_values[int(m / 10)]
                    output_gt.append(final_value)

        # Read the image and apply the transforms
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)[:, :, ::-1]
        transformed_img = self.transforms(image=img)["image"]

        return {
            "x": transformed_img,
            "y": torch.tensor(output_gt),
            "datetime": datetime_str,
            "img_path": img_path
        }

    @staticmethod
    def interpolate_values(x, y, num_values=7):
        return np.linspace(x, y, num_values).tolist()




