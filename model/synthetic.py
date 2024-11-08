import torch
import torch.nn as nn

from . import ModelBaseClass


class Synthetic(ModelBaseClass):
    def __init__(self, optim_cfg):
        super().__init__(optim_cfg)
        self.criterion = nn.BCEWithLogitsLoss()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

    def with_embedding(self, x):
        return x, self.linear(x)

    @property
    def fixed_embedding(self):
        return True

    def compute_loss(self, y_hat, y):
        return self.criterion(y_hat.squeeze(), y.float())

    def predict_step(self, batch):
        batch = batch.to(self.device)
        pred = self.forward(batch)
        return torch.sigmoid(pred)

    @staticmethod
    def compute_gradient(logits, _, embed):
        batchProbs = torch.sigmoid(logits)
        g = embed * batchProbs * (1 - batchProbs)
        return g


def get_model(_, optim_cfg):
    return Synthetic(optim_cfg)
