from abc import ABC
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class MyDataAugmentation(A.ImageOnlyTransform, ABC):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        # Perform the data augmentation here
        # For example, you might use the albumentations library's
        # Rotate, Flip, or ShiftScaleRotate transforms
        return augmented_image
