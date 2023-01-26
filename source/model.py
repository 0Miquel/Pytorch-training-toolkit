import torch.nn as nn
from torchvision import models
from abc import ABC, abstractmethod


def get_model(config):
    model_name = config['model_name']
    settings = config['settings']

    try:
        model = globals()[model_name](settings)
    except KeyError:
        raise f"Model with name {model_name} not found"

    return model


class MyModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super(MyModel, self).__init__()
        pass

    def freeze_model(self, fine_tune):
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False

    @abstractmethod
    def forward(self, x):
        pass


class Efficientnetb1(MyModel):
    def __init__(self, settings):
        super(Efficientnetb1, self).__init__()
        self.model = models.efficientnet_b1(pretrained=settings["pretrained"])
        self.freeze_model(settings["fine_tune"])
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=in_features, out_features=settings["n_classes"], bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet50(MyModel):
    def __init__(self, settings):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=settings["pretrained"])
        self.freeze_model(settings["fine_tune"])
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=settings["n_classes"], bias=True)

    def forward(self, x):
        x = self.model(x)
        return x
