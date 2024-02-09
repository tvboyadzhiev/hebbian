import torch
from torch import nn
import timm

class ImageClassifier(nn.Module):
    def __init__(self, network_name, classes, pretrained=True) -> None:
        super().__init__()

        self.latent_map = timm.create_model(network_name, pretrained=pretrained, features_only=True, out_indices=[-1])
        self.__latent_channels = self.latent_map(torch.randn(1, *self.latent_map.default_cfg['input_size']))[0].size()[-3]
        self.classify_head = nn.Linear(in_features=self.__latent_channels, out_features=classes)

    def forward(self, x):
        return self.classify_head(torch.mean(self.latent_map(x)[0], dim=[-2, -1]))

    @property
    def latent_channels(self):
        return self.__latent_channels
