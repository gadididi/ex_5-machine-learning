import sys
import torch
import numpy as np
import torchvision
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cnn import Net

from gcommand_dataset import GCommandLoader

EPOCHS = 10
LR = 0.07
MODEL = Net()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR)


def train(train_loader):
    MODEL.train()
    losses = 0
    correct = 0
    # getting the training set
    for batch_idx, (data_, labels) in enumerate(train_loader):
        OPTIMIZER.zero_grad()
        output_train = MODEL(data_)
        loss = F.nll_loss(output_train, labels)
        # losses += F.nll_loss(output_train, labels, reduction="mean").item()
        pred = output_train.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        loss.backward()
        MODEL.OPTIMIZER.step()


def test(val_loader):
    MODEL.eval()
    test_loss = 0
    tmp_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, target in val_loader:
            output = MODEL(data_)
            # tmp_loss += F.nll_loss(output, target, reduction="mean").item()
            test_loss += F.nll_loss(output, target, reduction="mean").item()
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
