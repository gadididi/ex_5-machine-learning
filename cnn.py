import torch.nn as nn
import torch
import torch.nn.functional
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            # first
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=1),

            # second
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=1),

            # third
            nn.Conv2d(32, 96, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=1),

            # four
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # five
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),


            # five
            nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # six
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear_layers = nn.Sequential(
            # first
            nn.Linear(3840, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            # second
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            # third
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            # four
            nn.Linear(128, 30),

        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
        self.criterion = nn.CrossEntropyLoss()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)


# import torch.nn as nn
# import torch
# import torch.nn.functional
# import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.cnn_layers = nn.Sequential(
#             # first
#             nn.Conv2d(1, 6, kernel_size=5),
#             nn.ReLU(),
#             nn.BatchNorm2d(6),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             # second
#             nn.Conv2d(6, 12, kernel_size=5),
#             nn.ReLU(),
#             nn.BatchNorm2d(12),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             # third
#             nn.Conv2d(12, 16, kernel_size=5),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#
#         self.linear_layers = nn.Sequential(
#             # first
#             nn.Linear(1920, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#
#             # second
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#
#             # third
#             nn.Linear(128, 30)
#         )
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
#         self.criterion = nn.CrossEntropyLoss()
#
#     # Defining the forward pass
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return F.log_softmax(x, dim=1)
