import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNet_ResNet50_FPN_V2_Weights


class RetinaNet(nn.Module):
    def __init__(self, n_classes):
        super(RetinaNet, self).__init__()
        self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

        # Modify the classification head
        # The original number of classes is model.head.classification_head.cls_logits.out_channels
        in_features = self.model.head.classification_head.conv[0][0].in_channels
        num_anchors = self.model.head.classification_head.num_anchors

        # Replace the classification head with a new one
        self.model.head = RetinaNetHead(in_features, num_anchors, n_classes)

    def forward(self, x, targets=None):
        if targets is None:
            x = self.model(x)
        else:
            x = self.model(x, targets)
        return x
