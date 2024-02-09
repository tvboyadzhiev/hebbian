import torch
import numpy as np

import config


class Accuracy:
    def __call__(self, predicted_probs: torch.Tensor, target_ids: torch.Tensor):
        return torch.mean(torch.argmax(predicted_probs, dim=1).eq(target_ids).to(torch.float32))

    def __str__(self):
        return "Accuracy"


class MeanIOU:
    def __init__(self, class_ids: torch.Tensor, ignore_index=-100, epsilon: float = 0.0001):
        self.__class_ids = class_ids.unsqueeze(0).unsqueeze(-1).unsqueeze(-2).to(config.device)
        self.__epsilon = epsilon
        self.__ignore_index = ignore_index

    def __call__(self, predicted_probs, target_ids):
        ignore_mask = target_ids.unsqueeze(1) != self.__ignore_index

        target = target_ids.unsqueeze(1).eq(self.__class_ids)
        predicted = torch.argmax(predicted_probs, dim=-3, keepdim=True).eq(self.__class_ids)

        intersection = torch.sum(predicted & target & ignore_mask, (-2, -1)) + self.__epsilon
        union = torch.sum((predicted | target) & ignore_mask, (-2, -1)) + self.__epsilon

        return torch.mean(
            intersection / union
        )

    def __str__(self):
        return "IoU"
