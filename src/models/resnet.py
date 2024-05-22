import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class Resnet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18', fine_tune=True):
        super(Resnet, self).__init__()
        if backbone == 'resnet18':
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        freeze_model(fine_tune, self.model)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


def freeze_model(fine_tune, model):
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False


class ResNet3D(nn.Module):
    def __init__(self, n_classes):
        super(ResNet3D, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        in_features = self.model.blocks[-1].proj.out_features
        self.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)

    def forward(self, x):
        x = x.permute((0, 2, 1, 3, 4))  # swap T and C
        out = self.model(x)
        out = self.fc(out)
        return out
