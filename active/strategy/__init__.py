from abc import ABC, abstractmethod
import torch
import numpy as np
from configs.strategy import StrategyConfig
from model import ModelBaseClass


class StrategyBaseClass(ABC):
    def __init__(self, model: ModelBaseClass, seed, num_samples, **kwargs):
        self.model = model
        self.num_samples = num_samples
        self.rnd = np.random.RandomState(seed)
        self.seed = seed

    @abstractmethod
    def __call__(self, trainer, unlabelled_pool):
        pass

    def state_dict(self):
        return {}


def to_multiclass_prob(probs: torch.Tensor):
    if probs.shape[1] == 1:
        probs = torch.hstack([1 - probs, probs])
    return probs


from .badge import BADGE  # type: ignore # noqa
from .disagree import Disagreement  # type: ignore # noqa
from .entropy import Entropy  # type: ignore # noqa
from .kmeans import KMeans  # type: ignore # noqa
from .margin import Margin  # type: ignore # noqa
from .oracle import Oracle  # type: ignore # noqa
from .random import Random  # type: ignore # noqa


class Strategy:
    def __init__(
        self, strategy_config: StrategyConfig, model, seed, num_samples, **kwargs
    ) -> None:
        self._strategy = eval(str(strategy_config["name"]))(
            model,
            seed,
            num_samples,
            **kwargs,
            **strategy_config,
        )

    def get(self):
        return self._strategy
