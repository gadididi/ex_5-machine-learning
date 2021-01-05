import torch
import torch.utils.data as data
import numpy as np

from gcommand_dataset import GCommandLoader


def main():
    train_set = GCommandLoader("./gcommands/train")
    val_set = GCommandLoader("./gcommands/valid")
    test_set = GCommandLoader("./gcommands/test")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True,
        pin_memory=True)


if __name__ == '__main__':
    main()
