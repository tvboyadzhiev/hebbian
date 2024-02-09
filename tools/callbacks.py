from pathlib import Path
from typing import Any
import torch

from torch.utils.tensorboard import SummaryWriter


class Checkpoint:
    def __init__(self, exp_root: Path):
        self.__exp_root = exp_root

    def save(self, model, opt, e, train_meas, val_meas, lr_scheduler, checkpoint_name):
        torch.save({
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,

            'epoch': e,
            'train_meas': train_meas,
            'val_meas': val_meas
        }, self.__exp_root / checkpoint_name)


class Last(Checkpoint):
    def __init__(self, exp_root: Path):
        super().__init__(exp_root)

    def __call__(self, model, opt, e, train_meas, val_meas, lr_scheduler):
        self.save(model, opt, e, train_meas, val_meas, lr_scheduler, 'last.pth')


class BestValidationLoss(Checkpoint):
    def __init__(self, exp_root: Path):
        super().__init__(exp_root)
        self.__best = None

    def __call__(self, model, opt, e, train_meas, val_meas, lr_scheduler):
        if self.__best is None or val_meas[0].last < self.__best:
            self.__best = val_meas[0].last
            self.save(model, opt, e, train_meas, val_meas, lr_scheduler, 'best_val_loss.pth')


class PeriodicCheckpoint(Checkpoint):
    def __init__(self, exp_root: Path, frequency:int):
        super().__init__(exp_root)
        self.__frequency = frequency

    def __call__(self, model, opt, e, train_meas, val_meas, lr_scheduler):
        if not e % self.__frequency:
            self.save(model, opt, e, train_meas, val_meas, lr_scheduler, f'epoch_{e:05d}.pth')
        

class Logger:
    def __init__(self, exp_root: Path):
        self.__file_name = exp_root / 'log.csv'

    def __call__(self, model, opt, e, train_meas, val_meas, lr_scheduler):
        if not self.__file_name.exists():
            with open(self.__file_name, 'w') as f:
                f.write("epoch, " +
                        ", ".join([str(m.name) for m in train_meas]) +
                        ", " +
                        ", ".join([str(m.name) for m in val_meas]) +
                        '\n')

        with open(self.__file_name, 'a') as f:
            f.write(f'{e:09d}, ' +
                    ", ".join([f'{m.last:.4f}' for m in train_meas]) +
                    ", " +
                    ", ".join([f'{m.last:.4f}' for m in val_meas]) +
                    '\n')


class SummaryWriterLogger:
    def __init__(self, exp_root: Path):
        self.__writer = SummaryWriter(exp_root / 'tensorboard')

    def __call__(self, model, opt, e, train_meas, val_meas, lr_scheduler):
        for m in train_meas:
            self.__writer.add_scalar(m.name, m.last, e)

        for m in val_meas:
            self.__writer.add_scalar(m.name, m.last, e)

    def __del__(self):
        self.__writer.close()
