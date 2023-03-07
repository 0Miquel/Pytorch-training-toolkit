import torch.nn as nn
import torch
from torchvision import models
import segmentation_models_pytorch as smp
from abc import ABC, abstractmethod


class Resnet50(nn.Module):
    def __init__(self, settings):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=settings["pretrained"])
        self.freeze_model(settings["fine_tune"])
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=settings["n_classes"], bias=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze_model(self, fine_tune):
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False
