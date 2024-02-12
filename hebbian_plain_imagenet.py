import os

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch import nn

import config
from tools import *
import tools.helpers as H
from models.layers import hebb


epochs = 300
batch_size = 32
levels = 4

def filters(level):
    if level < 0:
        return 3
    return 32 * 2**level

def stride(level):
    if level == 0:
        return 4
    return 3

train = DataLoader(
    ImageFolder(
        config.data_root / 'imagenet'/ 'train',
        transform=v2.Compose([
            v2.RandomResize(320, 600),
            v2.RandomCrop(320),
            v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count()
)

model = nn.Sequential(
    *[
        nn.Sequential(
            hebb.HebbianConv2d(filters(level-1), filters(level), kernel_size=stride(level)*2 - 1, stride=stride(level), alpha=1, padding=stride(level)-1),
            hebb.HebbianConv2d(filters(level), filters(level), 3, alpha=1, padding=1)
        )
        for level in range(levels)
    ] # kernel_size - 7 3 3 3; stride - 4 2 2 2; filters - 32 64 128 256
) # output is 1/32 of original resolution


opt = torch.optim.Adam(model.parameters())

if __name__ == '__main__':
    exp_root = H.build_exp_path(__file__)

    tools.train(model, train, opt,
        epochs=epochs,
        epoch_callbacks=[
            callbacks.Last(exp_root),  # Saves last state. Can be used to restart traning in case of crash
            callbacks.Logger(exp_root),  # Simple log file which can be seen with tail -f log.csv
        ],
        recover_checkpoint= exp_root / 'last.pth'
    )
