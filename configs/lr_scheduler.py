from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from omegaconf import MISSING


class LRScheduler(Enum):
    ExponentialLR = auto()
    StepLR = auto()

    def __str__(self):
        return self.name


@dataclass
class LRSchedulerConfig:
    name: Optional[LRScheduler] = None
    verbose: bool = MISSING


@dataclass
class ExponentialLR(LRSchedulerConfig):
    name = LRScheduler.ExponentialLR
    gamma: float = MISSING


@dataclass
class StepLR(LRSchedulerConfig):
    name = LRScheduler.StepLR
    step_size: int = MISSING
    gamma: float = MISSING
