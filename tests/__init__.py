import torch
from data import DataBaseClass


class TrainData(DataBaseClass):
    def __init__(self) -> None:
        self._y = torch.randint(0, 2, (len(self),))
        self._s = torch.randint(0, 2, (len(self),))

    def __getitem__(self, index):
        return index, (
            torch.rand(10),
            torch.randint(0, 3, (1,)).squeeze(),
            torch.randint(0, 3, (1,)).squeeze(),
        )

    def __len__(self):
        return 1000

    @property
    def x(self):
        pass

    @property
    def y(self):
        return self._y

    @property
    def s(self):
        return self._s

    @x.setter
    def x(self, x):
        pass

    @y.setter
    def y(self, y):
        pass

    @s.setter
    def s(self, s):
        pass


class TestData(TrainData):
    def __getitem__(self, idx):
        return idx, (
            torch.rand(10),
            torch.randint(0, 3, (1,)).squeeze(),
            torch.randint(0, 3, (1,)).squeeze(),
        )
