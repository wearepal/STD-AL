import numpy as np
import torch
from active.data import ActivePool
from configs.strategy import Divergence as DivergenceEnum
from model import CustomTrainer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import StrategyBaseClass, to_multiclass_prob


class Disagreement(StrategyBaseClass):
    def __init__(
        self,
        model,
        seed,
        num_samples,
        active_set: ActivePool,
        divergence=DivergenceEnum.KL,
        stochastic=False,
        **kwargs,
    ):
        super().__init__(model, seed, num_samples)
        self.active_set = active_set
        self.divergence = Divergence(name=divergence)
        self.stochastic = stochastic

    def get_iw(self):
        ys = self.active_set.labelled_pool.y.numpy()
        ss = self.active_set.labelled_pool.s.numpy()
        w = np.zeros_like(ys, dtype=float)
        self._state = dict()
        for y in np.unique(ys):
            for s in np.unique(ss):
                inds = np.where((ys == y) & (ss == s))[0]
                iw = (sum(ys == y) * sum(ss == s)) / (len(inds) * len(ys))
                w[inds] = iw
                self._state[f"strategy/iw/y={y.item()}/s={s.item()}"] = iw
        return torch.from_numpy(w)

    def __call__(self, trainer: CustomTrainer, w, loader: DataLoader, **kwargs):
        probs = trainer.predict(self.model, loader)
        probs = to_multiclass_prob(probs)
        self.train_weighted_model(trainer, w, loader.batch_size, loader.num_workers)
        weighted_probs = trainer.predict(self.model, loader)
        weighted_probs = to_multiclass_prob(weighted_probs)

        dist = self.divergence(probs, weighted_probs)
        self._state["strategy/div"] = dist.numpy()

        if not self.stochastic:
            return dist.sort(descending=True)[1].numpy()[: self.num_samples].tolist()
        else:
            p = (dist / dist.sum()).numpy()
            return self.rnd.choice(
                np.arange(len(p)),
                size=self.num_samples,
                p=p,
                replace=False,
            ).tolist()

    def train_weighted_model(self, trainer, initial_weights, batch_size, num_workers):
        self.model.load_state_dict(initial_weights)
        weights = self.get_iw()
        loader = DataLoader(
            self.active_set.labelled_pool,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        trainer.fit(self.model, loader, weights=weights)

    def state_dict(self):
        return self._state


class Divergence:
    def __init__(self, name: DivergenceEnum) -> None:
        self.__caller = {
            DivergenceEnum.KL: self.kl,
            DivergenceEnum.ReversedKL: self.reversed_kl,
            DivergenceEnum.Hellinger: self.hellinger,
            DivergenceEnum.Jeffreys: self.jeffreys,
            DivergenceEnum.Exponential: self.exponential,
        }[name]

    def __call__(self, p, q):
        div = self.__caller(p, q)
        return torch.clamp_min_(div, min=0)

    @staticmethod
    def kl(p, q):
        return F.kl_div(p.log(), q, reduction="none").sum(axis=1)

    @staticmethod
    def reversed_kl(p, q):
        return F.kl_div(q.log(), p, reduction="none").sum(axis=1)

    @staticmethod
    def hellinger(p, q):
        return 2 * ((p.sqrt() - q.sqrt()) ** 2).sum(1)

    @staticmethod
    def jeffreys(p, q):
        return ((p - q) * (p.log() - q.log())).sum(1)

    @staticmethod
    def exponential(p, q):
        return (p * (p.log() - q.log()) ** 2).sum(1)
