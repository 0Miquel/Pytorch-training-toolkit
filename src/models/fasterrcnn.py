import torch.nn as nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights


class FasterRCNN(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(FasterRCNN, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    def forward(self, x, targets=None):
        if targets is None:
            x = self.model(x)
        else:
            x = self.model(x, targets)
        return x
