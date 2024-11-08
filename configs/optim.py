from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import MISSING
from enum import Enum, auto


class Optim(Enum):
    SGD = auto()
    Adam = auto()

    def __str__(self):
        return self.name


@dataclass
class OptimConfig:
    name: Optim
    lr: float = MISSING


@dataclass
class SGD(OptimConfig):
    name = Optim.SGD
    momentum: Optional[float] = MISSING
    weight_decay: Optional[float] = MISSING
    dampening: Optional[float] = MISSING
    nesterov: Optional[bool] = MISSING


@dataclass
class Adam(OptimConfig):
    name = Optim.Adam
    betas: Optional[Tuple[float, float]] = MISSING
    eps: Optional[float] = MISSING
    weight_decay: Optional[float] = MISSING
    amsgrad: Optional[bool] = MISSING
