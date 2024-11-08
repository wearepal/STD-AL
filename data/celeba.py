from typing import Callable, Optional

import numpy as np
import torch
from configs.config import CelebAConfig
from ethicml.data.vision_data.celeba import celeba as celeba_
from ethicml.utility import DataTuple
from PIL import Image
from torchvision import transforms

from . import DataBaseClass


class CelebA(DataBaseClass):
    def __init__(
        self,
        root: str,
        label,
        sens_attr,
        seed: int,
        max_points: Optional[int] = None,
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
        dataset, self.base_dir = celeba_(self.root, label, sens_attr, download=True)
        dataset = dataset.load()

        # Load partially
        if max_points is not None:
            rnd = np.random.RandomState(seed)
            num_points = min(max_points, len(dataset))
            inds = rnd.choice(len(dataset), num_points, replace=False)
            dataset = DataTuple(
                dataset.x.iloc[inds],
                dataset.s.iloc[inds],
                dataset.y.iloc[inds],
            )

        self.__x = dataset.x["filename"].to_numpy()
        self.__y = torch.as_tensor(dataset.y.to_numpy()).squeeze()
        self.__s = torch.as_tensor(dataset.s.to_numpy()).squeeze()

    def __getitem__(self, index: int):
        x = Image.open(str(self.base_dir / self.x[index]))
        y, s = self.y[index], self.s[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return index, (x, y, s)

    def __len__(self) -> int:
        return self.y.size(0)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def s(self):
        return self.__s


def get_data(root, cfg: CelebAConfig):
    return CelebA(
        root,
        cfg.label,
        cfg.sens_attr,
        cfg.seed,
        cfg.max_points,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((cfg.size, cfg.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.mean, std=cfg.std),
            ]
        ),
    )
