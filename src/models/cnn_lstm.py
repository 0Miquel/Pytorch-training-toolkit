import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(self, n_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[:-1]))  # remove last layer
        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64, n_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2
