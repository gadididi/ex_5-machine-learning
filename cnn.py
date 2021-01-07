import torch.nn as nn
import torch
import torch.nn.functional
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.cnn_layers = nn.Sequential(
            # TODO: Change the sizes
            # TODO: Do we need BatchNorm2D?
            nn.Conv2d(1, 32, kernel_size=3, stride=4, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=2, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # TODO: Change the sizes
            nn.Linear(self.image_size, 100),
            nn.BatchNorm1d(100),  # applying batch norm
            nn.ReLU(),
            nn.Linear(100, 30)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
        self.criterion = nn.CrossEntropyLoss()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 37 * 22, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 37 * 22)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
