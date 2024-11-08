import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from active.strategy import Strategy
from active.strategy.disagree import Divergence, DivergenceEnum
from configs.strategy import Strategy as StrategyEnum
from data import CustomSubset, DataBaseClass
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .test_active_pool import ActivePool
from .test_model import NN, CustomTrainer, optim_cfg


class TestData(DataBaseClass):
    def __getitem__(self, idx):
        return idx, (torch.rand(10), self.y[idx], self.s[idx])

    def __len__(self):
        return 1000

    @property
    def classes(self):
        return torch.tensor([0, 1])

    @property
    def y(self):
        return torch.randint(0, len(self.classes), (len(self),))

    @property
    def x(self):
        pass

    @property
    def s(self):
        return torch.randint(0, len(self.classes), (len(self),))


class BinaryTestData(TestData):
    def __getitem__(self, idx):
        return idx, (torch.rand(2), self.y[idx], self.s[idx])

    @property
    def classes(self):
        return torch.tensor([0, 1])


class BinaryNN(NN):
    def __init__(self, optim_cfg):
        super().__init__(optim_cfg)
        self.criterion = nn.BCEWithLogitsLoss()
        self.linear = nn.Linear(2, 1)

    @property
    def dim(self):
        return 2

    def forward(self, x):
        return self.linear(x)

    def with_embedding(self, x):
        return x, self.linear(x)

    @staticmethod
    def compute_gradient(logits, _, embed):
        batchProbs = torch.sigmoid(logits)
        g = embed * batchProbs * (1 - batchProbs)
        return g

    def compute_loss(self, y_hat, y):
        return self.criterion(y_hat.squeeze(), y.float())


@pytest.mark.parametrize(
    "model, dataset",
    [
        (NN(optim_cfg), TestData("")),
        (BinaryNN(optim_cfg), BinaryTestData("")),
    ],
)
@pytest.mark.parametrize("s", StrategyEnum)
def test_strategy(model, dataset, s):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    subset = CustomSubset(dataset, torch.arange(0, 500))
    active_pool = ActivePool(subset, 0)
    active_pool.randomly_label(100)

    assert len(active_pool.unlabelled_pool) == 400

    query_size = 10
    strategy = Strategy(
        {"name": s},
        model,
        0,
        query_size,
        fixed_embedding=True,
        active_set=active_pool,
        classes=dataset.classes,
    ).get()
    trainer = CustomTrainer(num_epochs=1)

    loader = DataLoader(active_pool.unlabelled_pool, batch_size=32, num_workers=0)
    chosen = strategy(trainer=trainer, loader=loader, w=model.state_dict())
    assert isinstance(chosen, list)
    assert len(chosen) == query_size

    chosen = strategy(
        trainer=trainer, loader=loader, classes=dataset.classes, w=model.state_dict()
    )
    active_pool.label(chosen)
    assert isinstance(chosen, list)
    assert len(chosen) == query_size


@pytest.mark.parametrize("div", DivergenceEnum)
@pytest.mark.parametrize("sto", [True, False])
def test_disagreement(div, sto):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    model = NN(optim_cfg)
    dataset = TestData("")
    subset = CustomSubset(dataset, torch.arange(0, 500))
    active_pool = ActivePool(subset, 0)
    active_pool.randomly_label(100)

    assert len(active_pool.unlabelled_pool) == 400

    query_size = 10
    strategy = Strategy(
        {"name": StrategyEnum.Disagreement, "stochastic": sto},
        model,
        0,
        query_size,
        fixed_embedding=True,
        active_set=active_pool,
        classes=dataset.classes,
    ).get()
    trainer = CustomTrainer(num_epochs=1)

    loader = DataLoader(active_pool.unlabelled_pool, batch_size=32, num_workers=0)
    chosen = strategy(trainer=trainer, loader=loader, w=model.state_dict())
    assert isinstance(chosen, list)
    assert len(chosen) == query_size

    chosen = strategy(
        trainer=trainer, loader=loader, classes=dataset.classes, w=model.state_dict()
    )
    active_pool.label(chosen)
    assert isinstance(chosen, list)
    assert len(chosen) == query_size


@pytest.mark.parametrize("div", DivergenceEnum)
@pytest.mark.parametrize("d", [2, 10])
def test_divergence(div, d):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    for _ in range(100):
        p = F.softmax(torch.rand(100, d), 1)
        q = F.softmax(torch.rand(100, d), 1)

        divergence = Divergence(div)
        assert divergence(p, q).shape == torch.Size([100])
        assert sum(divergence(p, q) >= 0) == 100
