import torch.nn as nn
import torch
from torchvision import models
import segmentation_models_pytorch as smp
from abc import ABC, abstractmethod


class Unet(nn.Module):
    def __init__(self, settings):
        super(Unet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=settings["n_classes"],  # model output channels (number of classes in your data)
        )

    def forward(self, x):
        x = self.model(x)
        return x
