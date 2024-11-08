import importlib
from abc import ABC, abstractmethod

import dacite
import numpy as np
import pandas as pd
import torch
import wandb
from configs.config import DatasetConfig, Distribution, SyntheticConfig
from omegaconf import OmegaConf
from torch.utils.data import Subset, random_split
from torchvision.datasets import VisionDataset


class DataBaseClass(VisionDataset, ABC):
    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    @property
    @abstractmethod
    def s(self):
        pass

    def get_y_subset(self):
        for y in torch.unique(self.y):
            indices = torch.where(self.y == y)[0]
            if len(indices) > 0:
                yield y, Subset(self, indices)

    def get_s_subset(self):
        for s in torch.unique(self.s):
            indices = torch.where(self.s == s)[0]
            if len(indices) > 0:
                yield s, Subset(self, indices)

    def get_y_s_subset(self):
        for y in torch.unique(self.y):
            for s in torch.unique(self.s):
                indices = torch.where((self.y == y) & (self.s == s))[0]
                if len(indices) > 0:
                    yield (y, s), Subset(self, indices)

    def _all_y(self):
        return torch.unique(self.y)

    def _all_s(self):
        return torch.unique(self.s)


class CustomSubset(Subset):
    def get_y_subset(self):
        for y in torch.unique(self.y):
            indices = torch.where(self.dataset.y == y)[0]
            indices = torch.from_numpy(np.intersect1d(indices, self.indices))
            if len(indices) > 0:
                yield y, Subset(self.dataset, indices)

    def get_s_subset(self):
        for s in torch.unique(self.s):
            indices = torch.where(self.dataset.s == s)[0]
            indices = torch.from_numpy(np.intersect1d(indices, self.indices))
            if len(indices) > 0:
                yield s, Subset(self.dataset, indices)

    def get_y_s_subset(self):
        for y in torch.unique(self.y):
            for s in torch.unique(self.s):
                indices = torch.where((self.dataset.y == y) & (self.dataset.s == s))[0]
                indices = torch.from_numpy(np.intersect1d(indices, self.indices))
                if len(indices) > 0:
                    yield (y, s), Subset(self.dataset, indices)

    @property
    def y(self):
        return self.dataset.y[self.indices]

    @property
    def s(self):
        return self.dataset.s[self.indices]


class Dataset:
    def __init__(self, path, dist, data_cfg: DatasetConfig, seed):
        self._dataset = importlib.import_module(f"data.{data_cfg.name}").get_data(
            path, data_cfg
        )
        self.data_cfg = data_cfg
        self.seed = seed
        self.dist: Distribution = dacite.from_dict(
            data_class=Distribution,
            data=OmegaConf.to_container(dist),
            config=dacite.Config(check_types=False),
        )

    def get(self):
        if self.data_cfg.name == SyntheticConfig.name:
            train, test = self._dataset
            classes = torch.unique(train.y)
            train = Subset(train, np.arange(len(train)).tolist())
            test = Subset(test, np.arange(len(test)).tolist())
        else:
            train, test = self.split(
                self._dataset, self.data_cfg.train_split, self.seed
            )
            classes = torch.unique(self._dataset.y)
            if self.dist.unlabelled:
                imbalanced(train, self.dist.unlabelled, self.seed)
            if self.dist.test:
                imbalanced(test, self.dist.test, self.seed)

        log(test, "test")
        return (
            classes,
            train,
            CustomSubset(test.dataset, test.indices),
        )

    @staticmethod
    def split(dataset, train_test_ratio, seed):
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        return random_split(
            dataset, [train_size, test_size], torch.Generator().manual_seed(seed)
        )


def log(dataset: Subset, t: str):
    d = CustomSubset(dataset.dataset, dataset.indices)
    dist = []
    for (y, s), sub in d.get_y_s_subset():
        dist.append([len(sub) / len(d), y.item(), s.item()])
    df = pd.DataFrame(dist, columns=["p", "y", "s"])

    import plotly.express as px

    fig = px.bar(df, x="y", y="p", color="s", barmode="group", range_y=[0, 1])
    fig.update_traces(marker_coloraxis=None)
    wandb.log({f"dataset/{t}": fig}, step=0)


# def imbalanced(dataset: Subset, dist, seed: int):
#     dist = torch.tensor(dist)
#     dist = dist / dist.sum()

#     d = torch.zeros(len(dataset))
#     ys = dataset.dataset.y[dataset.indices]
#     ss = dataset.dataset.s[dataset.indices]

#     for y in torch.unique(ys):
#         for s in torch.unique(ss):
#             d[torch.where((ys == y) & (ss == s))[0]] = dist[y][s]

#     p = torch.rand(len(dataset), generator=torch.Generator().manual_seed(seed))
#     selected = torch.where(p <= d)[0]
#     dataset.indices = torch.tensor(dataset.indices)[selected].numpy().tolist()


def imbalanced(dataset: Subset, dist, seed: int):
    dist = torch.tensor(dist)
    dist = dist / dist.sum()

    d = torch.zeros(len(dataset))
    ys = dataset.dataset.y[dataset.indices]
    ss = dataset.dataset.s[dataset.indices]

    for y in torch.unique(ys):
        for s in torch.unique(ss):
            inds = torch.where((ys == y) & (ss == s))[0]
            dist[y][s] = dist[y][s] / (len(inds) / len(dataset))

    dist = dist / dist.sum()
    for y in torch.unique(ys):
        for s in torch.unique(ss):
            inds = torch.where((ys == y) & (ss == s))[0]
            d[inds] = dist[y][s]

    p = torch.rand(len(dataset), generator=torch.Generator().manual_seed(seed))
    selected = torch.where(p <= d)[0]
    dataset.indices = torch.tensor(dataset.indices)[selected].numpy().tolist()


def _imbalanced(dataset: Subset, dist, seed: int):
    dist = torch.tensor(dist)
    dist = dist / dist.sum()

    ys = dataset.dataset.y[dataset.indices]
    ss = dataset.dataset.s[dataset.indices]
    y_set, s_set = torch.unique(ys), torch.unique(ss)
    assert len(dataset) == len(ys) == len(ss)

    available_sizes = torch.zeros_like(dist, dtype=int)
    for y in y_set:
        for s in s_set:
            available_sizes[y][s] = len(torch.where((ys == y) & (ss == s))[0])

    for total_size in range(len(dataset), 0, -1):
        sizes = (total_size * dist).int()
        if (sizes <= available_sizes).sum() == torch.numel(available_sizes):
            break

    g = torch.Generator().manual_seed(seed)
    selected = torch.zeros_like(ys, dtype=bool)
    for y in y_set:
        for s in s_set:
            inds = torch.where((ys == y) & (ss == s))[0]
            rand_idx = torch.randperm(len(inds), generator=g)[: sizes[y][s].item()]
            selected[inds[rand_idx]] = True
    dataset.indices = torch.tensor(dataset.indices)[selected].numpy().tolist()
