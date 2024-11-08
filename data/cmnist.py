from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from configs.config import CMNISTConfig
from PIL import Image
from torchvision import datasets, transforms

from . import DataBaseClass

datasets.MNIST.resources = [
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
]


@dataclass
class Colour:
    R: float
    G: float
    B: float


class CMNIST(DataBaseClass):
    def __init__(
        self,
        root: str,
        colours: List[Colour],
        seed: int,
        digits: Optional[List[int]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        mnist_train = datasets.MNIST(self.root, train=True, download=True)
        mnist_test = datasets.MNIST(self.root, train=False, download=True)

        x = torch.cat((mnist_train.data, mnist_test.data))
        y = torch.cat((mnist_train.targets, mnist_test.targets))

        if digits is not None:
            inds = self.filter_digits(y, digits)
            x = x[inds]
            y = y[inds]

            for n, i in enumerate(torch.unique(y)):
                y[y == i] = n

        self.__x = torch.empty((0), dtype=torch.uint8)
        self.__y = torch.empty((0), dtype=y.dtype)
        self.__s = torch.empty((0), dtype=y.dtype)
        for i, c in enumerate(colours):
            self.__x = torch.cat((self.__x, self.add_colours(x, c)), 0)
            self.__y = torch.cat((self.__y, y), 0)
            self.__s = torch.cat((self.__s, torch.ones_like(y) * i))

        # shuffle
        gen = torch.Generator().manual_seed(seed)
        rand_idx = torch.randperm(len(self.__x), generator=gen)
        self.__x = self.__x[rand_idx]
        self.__y = self.__y[rand_idx]
        self.__s = self.__s[rand_idx]

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, index: int):
        img, target, context = self.x[index], int(self.y[index]), int(self.s[index])

        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, (img, target, context)

    @staticmethod
    def add_colours(x, c: Colour):
        return torch.stack((x * c.R, x * c.G, x * c.B), -1).to(torch.uint8)

    @staticmethod
    def filter_digits(y, digits: List[int]):
        inds = []
        for i in digits:
            inds.append(torch.where(y == i)[0])
        inds = torch.cat(inds)
        return inds

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def s(self):
        return self.__s


def get_data(root, cfg: CMNISTConfig):
    return CMNIST(
        root,
        cfg.colours,
        cfg.seed,
        cfg.digits,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
