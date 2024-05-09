import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)

    def forward(self, i):
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.AvgPool2d(4)(x)
        x = x.view(i.shape[0], i.shape[1], -1)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(750, 100)
        self.fc = nn.Linear(100 * 50, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


x = torch.rand((64, 50, 3, 32, 32))
net_cnn = CNN()
net_lstm = LSTM()

features = net_cnn(x)
out = net_lstm(features)

print(x.shape)
print(features.shape)
print(out.shape)
