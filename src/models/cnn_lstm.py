import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(self, n_classes, lstm_layers=64, backbone='resnet18'):
        super(CNNLSTM, self).__init__()

        if backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
        elif backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[:-1]))  # remove last layer

        if isinstance(lstm_layers, list):
            num_layers = len(lstm_layers)
            out_features = lstm_layers[-1]
        else:
            num_layers = 1
            out_features = lstm_layers
        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_layers,
            num_layers=num_layers,
            batch_first=True)

        self.linear = nn.Linear(out_features, n_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2
