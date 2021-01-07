import torch.nn as nn
import torch
import torch.nn.functional


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.cnn_layers = nn.Sequential(
            # TODO: Change the sizes
            # TODO: Do we need BatchNorm2D?
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # TODO: Change the sizes
            nn.Linear(self.image_size, 1000),
            nn.BatchNorm1d(1000),  # applying batch norm
            nn.ReLU(),
            nn.Linear(1000, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 30)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
