import pytest
import torch
from data import CustomSubset, DataBaseClass, _imbalanced, imbalanced
from data.celeba import CelebA
from data.cmnist import CMNIST, Colour
from data.synthetic import Synthetic
from torch.utils.data import random_split

datapath = "/srv/galene0/shared/data/" if torch.cuda.is_available() else "/tmp"


class TestDataset(DataBaseClass):
    @property
    def x(self):
        return None

    @property
    def y(self):
        return torch.cat((torch.ones(10) * 0, torch.ones(20) * 1, torch.ones(30) * 2))

    @property
    def s(self):
        return torch.cat((torch.ones(30) * 0, torch.ones(30) * 1))


class ImbalanceDataset(DataBaseClass):
    @property
    def x(self):
        return None

    @property
    def y(self):
        return torch.cat(
            (
                torch.ones(100000) * 0,
                torch.ones(80000) * 1,
                torch.ones(60000) * 0,
                torch.ones(30000) * 1,
            )
        ).to(torch.int64)

    @property
    def s(self):
        return torch.cat((torch.ones(180000) * 0, torch.ones(90000) * 1)).to(
            torch.int64
        )

    def __len__(self) -> int:
        return len(self.y)


def test_base():
    dataset = TestDataset(".")

    y_subset = dataset.get_y_subset()
    y, subset = next(y_subset)
    assert y == 0
    assert len(subset) == 10
    y, subset = next(y_subset)
    assert y == 1
    assert len(subset) == 20
    y, subset = next(y_subset)
    assert y == 2
    assert len(subset) == 30

    s_subset = dataset.get_s_subset()
    s, subset = next(s_subset)
    assert s == 0
    assert len(subset) == 30
    s, subset = next(s_subset)
    assert s == 1
    assert len(subset) == 30

    y_s_subset = dataset.get_y_s_subset()
    (y, s), subset = next(y_s_subset)
    assert (y, s) == (0, 0)
    assert len(subset) == 10
    (y, s), subset = next(y_s_subset)
    assert (y, s) == (1, 0)
    assert len(subset) == 20
    (y, s), subset = next(y_s_subset)
    assert (y, s) == (2, 1)
    assert len(subset) == 30


def test_custom_subset():
    dataset = TestDataset(".")
    customsubset = CustomSubset(dataset, torch.tensor(list(range(20, 60))))

    assert (torch.unique(customsubset.y) == torch.tensor([1.0, 2.0])).all()
    assert (torch.unique(customsubset.s) == torch.tensor([0.0, 1.0])).all()

    y_subset = customsubset.get_y_subset()
    y, subset = next(y_subset)
    assert y == 1
    assert len(subset) == 10
    y, subset = next(y_subset)
    assert y == 2
    assert len(subset) == 30

    s_subset = customsubset.get_s_subset()
    s, subset = next(s_subset)
    assert s == 0
    assert len(subset) == 10
    s, subset = next(s_subset)
    assert s == 1
    assert len(subset) == 30

    y_s_subset = customsubset.get_y_s_subset()
    (y, s), subset = next(y_s_subset)
    assert (y, s) == (1, 0)
    assert len(subset) == 10
    (y, s), subset = next(y_s_subset)
    assert (y, s) == (2, 1)
    assert len(subset) == 30


def test_cmnist():
    colours = [Colour(0, 0, 1), Colour(1, 0, 0)]
    dataset = CMNIST(datapath, colours, seed=0)

    assert dataset.x.shape == torch.Size([70000 * len(colours), 28, 28, 3])
    assert dataset.y.shape == torch.Size([70000 * len(colours)])
    assert dataset.s.shape == torch.Size([70000 * len(colours)])

    assert dataset.x.dtype == torch.uint8
    assert dataset.y.dtype == torch.int64
    assert dataset.s.dtype == torch.int64

    assert len(torch.unique(dataset.y)) == 10
    assert len(torch.unique(dataset.s)) == len(colours)

    assert len(dataset) == 70000 * len(colours)
    assert len(dataset[0][1]) == 3


def test_cmnist_filter_digits():
    colours = [Colour(0, 0, 1), Colour(1, 0, 0)]
    digits = [2, 5, 6]
    dataset = CMNIST(datapath, colours, seed=0, digits=digits)

    assert dataset.x.shape < torch.Size([70000 * len(colours), 28, 28, 3])
    assert dataset.y.shape < torch.Size([70000 * len(colours)])
    assert dataset.s.shape < torch.Size([70000 * len(colours)])

    assert len(torch.unique(dataset.y)) == len(digits)
    assert len(torch.unique(dataset.s)) == len(colours)

    for i in range(len(digits)):
        assert i in dataset.y


def test_celeba():
    dataset = CelebA(datapath, seed=0, label="Smiling", sens_attr="Male")

    assert len(dataset.x) == len(dataset.y) == len(dataset.s) == len(dataset)
    assert len(torch.unique(dataset.y)) == 2
    assert len(torch.unique(dataset.s)) == 2
    assert len(dataset[0][1]) == 3


def test_celeba_max_points():
    dataset = CelebA(
        datapath, seed=0, label="Smiling", sens_attr="Male", max_points=1000
    )
    assert len(dataset.x) == len(dataset.y) == len(dataset.s) == len(dataset) == 1000


@pytest.mark.parametrize("im", [imbalanced, _imbalanced])
def test_imbalance(im):
    dataset = ImbalanceDataset(".")
    dataset, _ = random_split(
        dataset, [len(dataset), 0], torch.Generator().manual_seed(0)
    )
    dist = torch.tensor([[0.1, 0.2], [0.4, 0.3]])
    im(dataset, dist, 0)
    dataset = CustomSubset(dataset.dataset, dataset.indices)

    y_dist = dist.sum(1)
    for y, subset in dataset.get_y_subset():
        assert len(subset) / len(dataset) == pytest.approx(y_dist[y].item(), abs=0.01)

    s_dist = dist.sum(0)
    for s, subset in dataset.get_s_subset():
        assert len(subset) / len(dataset) == pytest.approx(s_dist[s].item(), abs=0.01)

    for (y, s), subset in dataset.get_y_s_subset():
        assert len(subset) / len(dataset) == pytest.approx(dist[y][s].item(), abs=0.01)


@pytest.mark.parametrize("s_p", [None, 0.5])
def test_synthetic(s_p):
    num_points = 200000
    mean = [[[4, 4], [-8, 2]], [[8, -2], [-4, -4]]]
    alpha = 0.1
    y = 0.5

    dataset = Synthetic(0, mean, alpha, y, num_points, s=s_p)
    assert len(dataset) == num_points

    subgroup_prop = (
        [[alpha, 1 - alpha], [1 - alpha, alpha]]
        if s_p is None
        else [[s_p, 1 - s_p], [1 - s_p, s_p]]
    )

    y_s_subset = dataset.get_y_s_subset()
    for y, props, ms in zip([0, 1], subgroup_prop, mean):
        for s, p, m in zip([0, 1], props, ms):
            (y_, s_), subset = next(y_s_subset)
            assert (y_, s_) == (y, s)
            assert len(subset) / len(dataset) == pytest.approx(p / 2, abs=0.01)

            x_exp = torch.mean(subset.dataset.x[subset.indices], 0)
            x_std = torch.std(subset.dataset.x[subset.indices], 0)
            assert x_exp == pytest.approx(m, abs=0.01)
            assert x_std == pytest.approx([alpha, alpha], abs=0.01)
