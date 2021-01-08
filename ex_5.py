import sys
import torch
import numpy as np
import torchvision
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cnn import Net
from gcommand_dataset import GCommandLoader

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
EPOCHS = 20
LR = 0.01
best_model = Net()
best_model.to(device)


def train(train_loader):
    best_model.train()
    losses = 0
    # getting the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        output_train = best_model(data)
        loss = F.nll_loss(output_train.squeeze(), target)
        best_model.optimizer.zero_grad()
        loss.backward()
        best_model.optimizer.step()


def test(val_loader):
    best_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = best_model(data)
            test_loss += best_model.criterion(output, target).item()
            # get index of the max log - probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))


def run_model(train_loader, val_loader):
    for e in range(1, EPOCHS + 1):
        print("epoch number: ", e)
        train(train_loader)
        test(val_loader)


def main():
    train_set = GCommandLoader("./short_train")
    val_set = GCommandLoader("./short_valid")
    # test_set = GCommandLoader("./gcommands/test")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=True,
        pin_memory=True)
    run_model(train_loader, val_loader)


if __name__ == '__main__':
    main()
