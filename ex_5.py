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
EPOCHS = 100
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


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(val_loader):
    best_model.eval()
    test_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = best_model(data)
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

    print(
        f"\n\tAccuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n")


def prediction(test_loader):
    best_model.eval()
    f = open("test_y", "w")
    i = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print to file
            for image, predict in zip(images, predicted):
                data = str(test_loader.dataset.spects[i][0].split("/")[5])
                i += 1
                f.write(data + ", " + str(predict.item()))
                f.write("\n")
    f.close()


def run_model(train_loader, val_loader, test_loader=None):
    for e in range(1, EPOCHS + 1):
        print("epoch number: ", e)
        train(train_loader)
        test(val_loader)
    if test_loader is not None:
        prediction(test_loader)


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
    # test_loader = torch.utils.data.DataLoader(test_set)
    run_model(train_loader, val_loader, test_loader=None)


if __name__ == '__main__':
    main()
