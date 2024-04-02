import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class Resnet50(nn.Module):
    def __init__(self, pretrained, fine_tune, n_classes):
        super(Resnet50, self).__init__()
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50()

        freeze_model(fine_tune, self.model)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet18(nn.Module):
    def __init__(self, pretrained, fine_tune, n_classes):
        super(Resnet18, self).__init__()
        if pretrained:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18()

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
