from dataclasses import dataclass
from enum import Enum, auto

from omegaconf import MISSING


class Strategy(Enum):
    Random = auto()
    BADGE = auto()
    Margin = auto()
    Entropy = auto()
    KMeans = auto()
    Oracle = auto()
    Disagreement = auto()

    def __str__(self):
        return self.name


@dataclass
class StrategyConfig:
    name: Strategy


@dataclass
class Random(StrategyConfig):
    name = Strategy.Random


@dataclass
class BADGE(StrategyConfig):
    name = Strategy.BADGE


@dataclass
class Margin(StrategyConfig):
    name = Strategy.Margin


@dataclass
class Entropy(StrategyConfig):
    name = Strategy.Entropy


@dataclass
class KMeans(StrategyConfig):
    name = Strategy.KMeans


@dataclass
class Oracle(StrategyConfig):
    name = Strategy.Oracle


class Divergence(Enum):
    KL = auto()
    ReversedKL = auto()
    Hellinger = auto()
    Jeffreys = auto()
    Exponential = auto()

    def __str__(self):
        return self.name


@dataclass
class Disagreement(StrategyConfig):
    name = Strategy.Disagreement
    divergence: Divergence = MISSING
    stochastic: bool = MISSING
