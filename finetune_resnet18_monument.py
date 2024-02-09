from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import v2

import config
from tools import *
from models import ImageClassifier


exp_root = config.exp_root / Path(__file__).stem
if not exp_root.exists():
    exp_root.mkdir(parents=True)

epochs = 300
num_classes = 12
batch_size = 32


train = DataLoader(
    ImageFolder(
        config.data_root / 'pisa-monuments' / 'train_test',
        transform=v2.Compose([
            v2.RandomResize(256, 480),
            v2.RandomCrop(224),
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
    num_workers=20
)

val = DataLoader(
    ImageFolder(
        config.data_root / 'pisa-monuments' / 'val',
        transform=v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=20
)


model = ImageClassifier('resnet18', classes=num_classes, pretrained=True)

opt = optim.AdamW(model.parameters(), lr=0.0001)
lr_schedule = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

m = [measurements.Accuracy()]

tools.train(
    model,
    torch.nn.CrossEntropyLoss(label_smoothing=0.1),
    train,
    val,
    opt=opt,
    epochs=epochs,
    train_measurements=m,
    val_measurements=m,
    epoch_callbacks=[
        callbacks.BestValidationLoss(exp_root),  # Saves the lowest loss
        callbacks.Last(exp_root),  # Saves last state. Can be used to restart traning in case of crash
        callbacks.Logger(exp_root),  # Simple log file which can be seen with tail -f log.csv
        callbacks.SummaryWriterLogger(exp_root)  # Tensorboard Logger
    ],
    learning_rate_schedule=lr_schedule,
    recover_checkpoint=exp_root / 'last.pth'
)
