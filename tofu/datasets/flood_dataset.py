from torch.utils.data import Dataset
import glob
import cv2
import os


class FloodAreaSegmentation(Dataset):
    def __init__(self, data_path, settings, transforms):
        images_path = os.path.join(data_path, settings["images_path"])
        masks_path = os.path.join(data_path, settings["masks_path"])

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

        return {"imgs": transformed_img, "masks": mask}
