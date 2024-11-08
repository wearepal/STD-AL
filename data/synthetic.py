from typing import List

import numpy as np
import torch
from configs.config import SyntheticConfig

from . import DataBaseClass

MEAN_TYPE = List[List[List[float]]]
A_TYPE = List[List[float]]


class Synthetic(DataBaseClass):
    def __init__(
        self,
        seed: int,
        mean: MEAN_TYPE,
        alpha: float,
        y: float,
        num_points,
        s=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(root="")
        rng = np.random.RandomState(seed)

        ysizes = [num_points - int(num_points * y), int(num_points * y)]
        # subgroup_std = [[alpha, 1 - alpha], [1 - alpha, alpha]]
        subgroup_prop = (
            [[alpha, 1 - alpha], [1 - alpha, alpha]]
            if s is None
            else [[s, 1 - s], [1 - s, s]]
        )

        xx, yy, ss = [], [], []
        for y_, (means, sub_props, yp) in enumerate(zip(mean, subgroup_prop, ysizes)):
            for s_, (m, sp) in enumerate(zip(means, sub_props)):
                x = rng.multivariate_normal(
                    np.array(m), (alpha ** 2) * np.eye(len(m)), int(sp * yp)
                )
                xx.append(x)
                ss.append(np.ones(int(sp * yp), dtype=int) * s_)
                yy.append(np.ones(int(sp * yp), dtype=int) * y_)

        self.__x = torch.from_numpy(np.vstack(xx))
        self.__y = torch.from_numpy(np.concatenate(yy))
        self.__s = torch.from_numpy(np.concatenate(ss))

        # shuffle
        gen = torch.Generator().manual_seed(seed)
        rand_idx = torch.randperm(len(self.__x), generator=gen)
        self.__x = self.__x[rand_idx]
        self.__y = self.__y[rand_idx]
        self.__s = self.__s[rand_idx]

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, index: int):
        x, target, context = self.x[index], int(self.y[index]), int(self.s[index])

        return index, (x.float(), target, context)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def s(self):
        return self.__s


def get_data(_, cfg: SyntheticConfig):
    train = Synthetic(
        cfg.seed,
        cfg.mean,
        cfg.train.alpha,
        cfg.train.y,
        cfg.train.num_points,
        cfg.train.s,
    )
    test = Synthetic(
        cfg.seed,
        cfg.mean,
        cfg.test.alpha,
        cfg.test.y,
        cfg.test.num_points,
        cfg.test.s,
    )
    return train, test
