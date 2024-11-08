import torch.nn as nn
from configs.config import CMNISTConfig

from . import ModelBaseClass


class CMNIST(ModelBaseClass):
    def __init__(self, num_classes, optim_cfg):
        super().__init__(optim_cfg)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 7 * 7 * 32)
        x = self.fc(x)
        return x

    def with_embedding(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        embed = x.view(-1, 7 * 7 * 32)
        x = self.fc(embed)
        return embed, x

    @property
    def fixed_embedding(self):
        return False


def get_model(cfg: CMNISTConfig, optim_cfg):
    return CMNIST(len(cfg.digits), optim_cfg)
