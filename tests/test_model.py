import pytest
import torch
import torch.nn as nn
from model import CustomTrainer, ModelBaseClass
from torch.utils.data import DataLoader, Dataset

from . import TestData, TrainData


class NN(ModelBaseClass):
    def __init__(self, optim_cfg):
        super().__init__(optim_cfg)
        self.layer1 = nn.Linear(10, self.dim)
        self.layer2 = nn.Linear(self.dim, 3)

    @property
    def dim(self):
        return 500

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def with_embedding(self, x):
        embed = self.layer1(x)
        out = self.layer2(embed)
        return embed, out


class InferenceData(Dataset):
    def __getitem__(self, idx):
        return torch.rand(10)

    def __len__(self):
        return 1000

    @property
    def classes(self):
        return torch.tensor([0, 1, 2])

    @property
    def y(self):
        return torch.randint(0, 3, len(self))


optim_cfg = {
    "name": "Adam",
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "amsgrad": False,
}


@pytest.mark.parametrize("clamp_grad", [[-0.1, 0.1], None])
@pytest.mark.parametrize(
    "lr_scheduler_config",
    [None, {"name": "StepLR", "verbose": False, "step_size": 1, "gamma": 0.8}],
)
def test_train(clamp_grad, lr_scheduler_config):
    num_epochs = 2
    model = NN(optim_cfg)
    loader = DataLoader(TrainData(), 10, num_workers=0)
    trainer = CustomTrainer(
        num_epochs=num_epochs,
        clamp_grad=clamp_grad,
        lr_scheduler_config=lr_scheduler_config,
    )
    state = trainer.fit(model, loader)
    assert len(model.train_history) == num_epochs
    if lr_scheduler_config is not None:
        assert state["lr_scheduler"]["_last_lr"][0] < optim_cfg["lr"]


def test_predict():
    model = NN(optim_cfg)
    batch_size = 4
    dataset = InferenceData()
    loader = DataLoader(dataset, batch_size, num_workers=0)
    trainer = CustomTrainer()
    predictions = trainer.predict(model, loader)
    assert predictions.shape == torch.Size([len(dataset), 3])


def test_test():
    model = NN(optim_cfg)
    loader = DataLoader(TestData(), 10, num_workers=0)
    trainer = CustomTrainer()
    trainer.test(model, loader)
    assert 0 <= model.metrics["test_acc"].compute() <= 1


def test_gradient_embeddings():
    model = NN(optim_cfg)
    dataset = InferenceData()
    loader = DataLoader(dataset, 10, num_workers=0)
    trainer = CustomTrainer()
    gradient_embeddings = trainer.gradient_embeddings(dataset.classes, model, loader)
    assert gradient_embeddings.shape == torch.Size(
        [len(dataset), len(dataset.classes) * model.dim]
    )


def test_gradient_embeddings_with_return_feature():
    model = NN(optim_cfg)
    dataset = InferenceData()
    loader = DataLoader(dataset, 10, num_workers=0)
    trainer = CustomTrainer()
    embeddings, gradient_embeddings = trainer.gradient_embeddings(
        dataset.classes, model, loader, return_embedding=True
    )
    assert gradient_embeddings.shape == torch.Size(
        [len(dataset), len(dataset.classes) * model.dim]
    )
    assert embeddings.shape == torch.Size([len(dataset), model.dim])


def test_embeddings():
    model = NN(optim_cfg)
    dataset = InferenceData()
    loader = DataLoader(dataset, 10, num_workers=0)
    trainer = CustomTrainer()
    gradient_embeddings = trainer.embeddings(model, loader)
    assert gradient_embeddings.shape == torch.Size([len(dataset), model.dim])
