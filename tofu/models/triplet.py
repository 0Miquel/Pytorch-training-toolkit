import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict


class EmbeddingNetImage(nn.Module):
    def __init__(self, features, pretrained=True):  # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features=features)

    def forward(self, x):
        output = self.model(x)

        return output


class TripletNet(nn.Module):
    def __init__(self, features, pretrained=True):
        super(TripletNet, self).__init__()
        self.embedding_net_image = EmbeddingNetImage(features, pretrained)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_image(x1)
        output2 = self.embedding_net_image(x2)
        output3 = self.embedding_net_image(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)
