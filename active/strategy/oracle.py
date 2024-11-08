import numpy as np
import torch
from active.data import ActivePool

from . import StrategyBaseClass


class Oracle(StrategyBaseClass):
    def __init__(self, model, seed, num_samples, active_set: ActivePool, **kwargs):
        super().__init__(model, seed, num_samples)
        self.active_set = active_set

    def __call__(self, *args, **kwargs):
        classes, ps = self.current_dist()
        yy = self.active_set.unlabelled_pool.y
        ss = self.active_set.unlabelled_pool.s

        n = []
        for i in np.argsort(ps):
            y, s = classes[i]
            inds = torch.where((yy == y) & (ss == s))[0].numpy()
            t = self.num_samples - len(n)
            if len(inds) > 0:
                if len(inds) < t:
                    n += inds.tolist()
                else:
                    n += self.rnd.choice(inds, t, replace=False).tolist()
            if len(n) == self.num_samples:
                break

        assert len(n) == self.num_samples
        return n

    def current_dist(self):
        yy = self.active_set.labelled_pool.y
        ss = self.active_set.labelled_pool.s

        classes, ps = [], []
        for y in torch.unique(yy):
            for s in torch.unique(ss):
                p = len(torch.where((yy == y) & (ss == s))[0]) / len(yy)
                classes.append((y, s))
                ps.append(p)

        return classes, ps
