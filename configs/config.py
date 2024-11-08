from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from omegaconf import MISSING

from .lr_scheduler import LRSchedulerConfig
from .optim import OptimConfig
from .strategy import StrategyConfig


class WandbMode(Enum):
    online = auto()
    offline = auto()
    disabled = auto()


@dataclass
class WandBConfig:
    mode: WandbMode = MISSING
    project: str = MISSING
    entity: str = MISSING


@dataclass
class DatasetConfig:
    name: str
    train_split: float = MISSING
    seed: int = MISSING


@dataclass
class Colour:
    R: float
    G: float
    B: float


@dataclass
class CMNISTConfig(DatasetConfig):
    name: str = "cmnist"
    digits: List[int] = MISSING
    colours: List[Colour] = MISSING


@dataclass
class CelebAConfig(DatasetConfig):
    name: str = "celeba"
    max_points: Optional[int] = MISSING
    label: str = MISSING
    sens_attr: str = MISSING
    size: int = MISSING
    mean: List[float] = MISSING
    std: List[float] = MISSING


@dataclass
class SyntheticProp:
    y: float
    s: Optional[float]
    alpha: float
    num_points: int


@dataclass
class SyntheticConfig(DatasetConfig):
    name: str = "synthetic"
    mean: List[float] = MISSING
    train: SyntheticProp = MISSING
    test: SyntheticProp = MISSING


@dataclass
class Distribution:
    unlabelled: Optional[List[float]]
    test: Optional[List[float]]
    initial: Optional[List[float]]


@dataclass
class ExperimentConfig:
    name: str
    num_epochs: int = MISSING
    num_steps: int = MISSING
    batch_size: int = MISSING
    initial: int = MISSING
    n_to_label: int = MISSING
    seed: int = MISSING
    dist: Distribution = MISSING
    clamp_grad: Optional[List[float]] = MISSING


@dataclass
class Config:
    data: DatasetConfig
    exp: ExperimentConfig
    optim: OptimConfig
    wandb: WandBConfig
    strategy: StrategyConfig
    lr_scheduler: LRSchedulerConfig
