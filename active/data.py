from typing import Union

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, Subset


class ActivePool(Dataset):
    def __init__(self, unlabelled_set: Subset, seed) -> None:
        self._dataset = unlabelled_set
        self._rnd = np.random.RandomState(seed)
        self._labelled = np.zeros(len(self._dataset), dtype=bool)
        self._history = []

    def randomly_label(self, n: int):
        indices = self._rnd.choice((~self._labelled).nonzero()[0], n, replace=False)
        self._history.append(indices.tolist())
        self._labelled[indices] = 1

    @property
    def unlabelled_pool(self):
        unlabelled = (~self._labelled).nonzero()[0]
        return Pool(self._dataset, unlabelled)

    @property
    def _true_labels(self):
        pool = self.unlabelled_pool
        return pool.y, pool.s

    @property
    def labelled_pool(self):
        labelled = self._labelled.nonzero()[0]
        return Pool(self._dataset, labelled, train=True)

    def __getitem__(self, index):
        return (
            index,
            self._dataset[self._labelled.nonzero()[0][index].squeeze().item()][1],
        )

    def __len__(self) -> int:
        return self._labelled.sum()

    def label(self, index: Union[list, int]):
        if isinstance(index, int):
            index = [index]

        lbl_nz = (~self._labelled).nonzero()[0]
        indexes = [int(lbl_nz[idx].squeeze().item()) for idx in index]
        self._history.append(indexes)
        self._labelled[indexes] = 1
        return Subset(self._dataset, indexes)

    @property
    def history(self):
        return self._history

    def labelled_state(self):
        state = dict()

        labelled = self._labelled.nonzero()[0]
        labelled = torch.tensor(self._dataset.indices)[labelled]

        ys = self._dataset.dataset.y[labelled]
        ss = self._dataset.dataset.s[labelled]

        for y in self._dataset.dataset._all_y():
            state[f"labelled/y={y.item()}"] = len(torch.where(ys == y)[0])

        for s in self._dataset.dataset._all_s():
            state[f"labelled/s={s.item()}"] = len(torch.where(ss == s)[0])

        sizes = []
        for y in self._dataset.dataset._all_y():
            for s in self._dataset.dataset._all_s():
                inds = torch.where((ys == y) & (ss == s))[0]
                state[f"labelled/y={y.item()}/s={s.item()}"] = len(inds)
                sizes.append(len(inds))
        p = np.array(sizes)
        p = p / p.sum()
        h = (-p * np.log2(p, out=np.zeros_like(p), where=(p != 0))).sum() / np.log2(
            len(p)
        )

        state["active/lambda_"] = h
        state["active/labelled"] = len(self)
        return state

    def unlabelled_state(self):
        state = dict()

        unlabelled = (~self._labelled).nonzero()[0]
        unlabelled = torch.tensor(self._dataset.indices)[unlabelled]

        ys = self._dataset.dataset.y[unlabelled]
        ss = self._dataset.dataset.s[unlabelled]

        for y in self._dataset.dataset._all_y():
            state[f"unlabelled/y={y.item()}"] = len(torch.where(ys == y)[0])

        for s in self._dataset.dataset._all_s():
            state[f"unlabelled/s={s.item()}"] = len(torch.where(ss == s)[0])

        for y in self._dataset.dataset._all_y():
            for s in self._dataset.dataset._all_s():
                inds = torch.where((ys == y) & (ss == s))[0]
                if len(inds) > 0:
                    state[f"unlabelled/y={y.item()}/s={s.item()}"] = len(inds)

        state["active/unlabelled"] = len((~self._labelled).nonzero()[0])
        return state

    @property
    def state(self):
        state = self.unlabelled_state()
        state.update(self.labelled_state())
        return state

    def label_based_on_prop(self, dist, n):
        dist = torch.tensor(dist)
        dist = dist / dist.sum()

        size = (n * dist).to(int)

        while size.sum() != n:
            i = self._rnd.randint(dist.shape[0])
            j = self._rnd.randint(dist.shape[1])
            size[i, j] += 1
        assert size.sum() == n

        unlabelled_inds = np.nonzero(~self._labelled)[0]
        inds = np.array(self._dataset.indices)[unlabelled_inds]

        ys = self._dataset.dataset.y[inds]
        ss = self._dataset.dataset.s[inds]

        indexes = []
        for y in torch.unique(ys):
            for s in torch.unique(ss):
                inds = torch.where((ys == y) & (ss == s))[0]
                t = self._rnd.permutation(len(inds))[: size[y, s]]
                inds = inds[t]
                self._labelled[unlabelled_inds[inds]] = 1
                indexes += unlabelled_inds[inds].tolist()
        self._history.append(indexes)

    def reset(self):
        self._labelled = np.zeros(len(self._dataset), dtype=bool)
        self._history = []

    def load_history(self, history):
        if isinstance(history, np.ndarray):
            history = history.tolist()
        self._history = history
        inds = np.concatenate(history)
        self._labelled[inds] = 1


class Pool(Subset):
    def __init__(self, dataset, indices, train=False) -> None:
        super().__init__(dataset, indices)
        self.train = train

    def __getitem__(self, idx):
        if not self.train:
            # only return x
            return self.dataset[self.indices[idx]][1][0]
        return idx, self.dataset[self.indices[idx]][1]

    @property
    def y(self):
        return self.dataset.dataset.y[np.array(self.dataset.indices)[self.indices]]

    @property
    def s(self):
        return self.dataset.dataset.s[np.array(self.dataset.indices)[self.indices]]

    @property
    def x(self):
        return self.dataset.dataset.x[np.array(self.dataset.indices)[self.indices]]


def log_labelled(active_pool: ActivePool, name, step=0):
    state = active_pool.labelled_state()
    length = state["active/labelled"]
    _log_active_pool(state, "labelled", length, name, step)


def log_unlabelled(active_pool: ActivePool, name, step=0):
    state = active_pool.unlabelled_state()
    length = state["active/unlabelled"]
    _log_active_pool(state, "unlabelled", length, name, step)


def _log_active_pool(state, g, length, name, step):
    import re
    import pandas as pd

    dist = []
    for key, val in state.items():
        t = re.findall(f"{g}/y=(.*)/s=(.*)", key)
        if not t:
            continue
        t = t[0]
        y, s = int(t[0]), int(t[1])
        dist.append([val / length, y, s])

    df = pd.DataFrame(dist, columns=["p", "y", "s"])

    import plotly.express as px

    fig = px.bar(df, x="y", y="p", color="s", barmode="group", range_y=[0, 1])
    fig.update_traces(marker_coloraxis=None)
    wandb.log({f"dataset/{name}": fig}, step=step)
