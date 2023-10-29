import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=20):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_channel=3, output_channel=12):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=output_channel, kernel_size=5),
        )

    def forward(self, x):
        # x : BxCxHxW
        x = self.layers(x)
        return x


class Net(nn.Module):
    def __init__(self, input_channel=3, output_channel=12):
        super(Net, self).__init__()
        self.cnn_layers = CNN(input_channel, output_channel)
        self.fc_layers = MLP(input_dim=250*250*12, output_dim=6)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten()
        x = self.fc_layers(x)
        return x