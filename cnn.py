import torch.nn as nn
import torch
import torch.nn.functional
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            # first
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # second
            nn.Conv2d(6, 12, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # third
            nn.Conv2d(12, 16, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # first
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            # second
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            # third
            nn.Linear(128, 30)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
        self.criterion = nn.CrossEntropyLoss()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)
