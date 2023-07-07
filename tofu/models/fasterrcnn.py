import torch.nn as nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    def __init__(self, settings):
        super(FasterRCNN, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=settings["pretrained"])
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, settings["n_classes"])

    def forward(self, x):
        x = self.model(x)
        return x
