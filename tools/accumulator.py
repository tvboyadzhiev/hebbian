from pathlib import Path
import numpy as np
import torch


class Accumulator:
    def __init__(self, name, measurement):
        self.__measurement = measurement
        self.__sum = 0.0
        self.__count = 0.0

        self.__history = []
        self.__name = name

    def __call__(self, *args):
        current_measurement = self.__measurement(*args)
        if isinstance(current_measurement, torch.Tensor):
            self.__sum += current_measurement.item()
        else:
            self.__sum += current_measurement
        self.__count += 1
        return current_measurement

    def end_epoch(self):
        self.__history.append(self.__sum/self.__count)
        self.__sum = 0.0
        self.__count = 0

    def save(self, root: Path):
        np.savetxt(root / f'{self.__name}.csv', np.array(self.__history), delimiter=",")

    @property
    def last(self):
        return self.__history[-1]

    @property
    def history(self):
        return self.__history.copy()

    @property
    def name(self):
        return self.__name
