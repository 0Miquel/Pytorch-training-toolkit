import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class Resnet50(nn.Module):
    def __init__(self, settings):
        super(Resnet50, self).__init__()
        if settings["pretrained"]:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50()

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


class Resnet18(nn.Module):
    def __init__(self, settings):
        super(Resnet18, self).__init__()
        if settings["pretrained"]:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18()

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
