import os

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import v2

import config
from tools import *
from models import ImageClassifier
import tools.helpers as H

def trail(exp_root):
    epochs = 300
    num_classes = 12
    batch_size = 32

    train_data_root, test_data_root = H.random_split(
        config.data_root / 'pisa-monuments' / 'train_test',
        exp_root
    )
    print(train_data_root, ' ', test_data_root)
    
    train = DataLoader(
        ImageFolder(
            train_data_root,
            transform=v2.Compose([
                v2.RandomResize(300, 450),
                v2.RandomCrop(300),
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

    val = DataLoader(
        ImageFolder(
            test_data_root,
            transform=v2.Compose([
                v2.Resize(300),
                v2.CenterCrop(300),
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
        num_workers=os.cpu_count()
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

if __name__ == '__main__':
    exp_root = H.build_exp_path(__file__)

    # repeat an experiment 
    for i in range(5):
        (exp_root / f'trail_{i}').mkdir(parents=True)
        trail(exp_root / f'trail_{i}')
