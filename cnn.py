import torch.nn as nn
import torch
import torch.nn.functional
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            # first
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=5, stride=1),

            # second
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # third
            nn.Conv2d(64, 40, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.linear_layers = nn.Sequential(
            # first
            nn.Linear(5120, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            # second
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            # third
            nn.Linear(128, 30)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)
