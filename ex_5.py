import sys
import torch
import numpy as np
import torchvision
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cnn import Net

from gcommand_dataset import GCommandLoader


EPOCHS = 100
LR = 0.01
best_model = Net()


def train(train_loader):
    best_model.train()
    losses = 0
    # getting the training set
    for batch_idx, (data_, labels) in enumerate(train_loader):
        best_model.optimizer.zero_grad()
        output_train = best_model(data_)
        loss = best_model.criterion(output_train, labels)
        loss.backward()
        best_model.optimizer.step()


def test(val_loader):
    best_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, target in val_loader:
            output = best_model(data_)
            test_loss += best_model.criterion(output, target).item()
            # get index of the max log - probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(val_loader.dataset)
    print("\nTests set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct,
                                                                                  len(val_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      val_loader.sampler)))


def run_model(train_loader, val_loader):
    for e in range(1, EPOCHS + 1):
        print("epoch number: ", e)
        train(train_loader)
        test(val_loader)


def main():
    train_set = GCommandLoader("./gcommands/train")
    val_set = GCommandLoader("./gcommands/valid")
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
