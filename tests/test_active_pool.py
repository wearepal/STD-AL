import torch
from active.data import ActivePool
from torch.utils.data import Subset

from . import TrainData
from .test_data import ImbalanceDataset
import pytest


def test_label_randomly():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(50))
    active_pool = ActivePool(dataset, 0)

    assert len(active_pool) == 0
    assert len(active_pool.unlabelled_pool) == 50

    active_pool.randomly_label(2)
    assert len(active_pool) == 2
    assert len(active_pool.unlabelled_pool) == 48
    assert (
        active_pool.unlabelled_pool.indices == (~active_pool._labelled).nonzero()
    ).all()


def test_reset():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(50))
    active_pool = ActivePool(dataset, 0)

    active_pool.randomly_label(10)
    assert len(active_pool.unlabelled_pool) == 40
    assert len(active_pool) == 10

    active_pool.reset()
    assert len(active_pool.unlabelled_pool) == 50
    assert len(active_pool) == 0
    assert active_pool.history == []


def test_load_history():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(100))
    active_pool = ActivePool(dataset, 0)

    history = [[0, 1, 2], [5, 6, 8, 10]]
    assert len(active_pool) == 0
    active_pool.load_history(history)
    assert len(active_pool) == 7
    assert len(active_pool.history) == 2


def test_states():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(500))
    active_pool = ActivePool(dataset, 0)

    active_pool.randomly_label(100)
    assert len(active_pool.unlabelled_state()) == 9
    assert len(active_pool.labelled_state()) == 10


def test_label():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(50))
    active_pool = ActivePool(dataset, 0)

    assert len(active_pool) == 0
    assert len(active_pool.unlabelled_pool) == 50

    active_pool.label(0)
    active_pool.label(4)
    active_pool.label(6)
    assert len(active_pool) == 3
    assert len(active_pool.unlabelled_pool) == 47

    assert active_pool._labelled[0]
    assert active_pool._labelled[5]
    assert active_pool._labelled[8]


def test_labelled_pool():
    dataset = ImbalanceDataset(".")
    dataset = Subset(dataset, torch.arange(0, 1000))
    assert len(dataset) == 1000

    active_pool = ActivePool(dataset, 0)
    active_pool.randomly_label(200)

    assert len(active_pool.labelled_pool) == 200
    assert len(active_pool.labelled_pool.y) == 200


def test_multiplelabel():
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(50))
    active_pool = ActivePool(dataset, 0)

    assert len(active_pool) == 0
    assert len(active_pool.unlabelled_pool) == 50

    subset = active_pool.label([0, 1, 2, 3, 4])
    assert len(active_pool) == 5
    assert len(subset) == 5
    assert len(active_pool.unlabelled_pool) == 45


def test_label_based_on_prop():
    torch.manual_seed(0)
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(200))
    active_pool = ActivePool(dataset, 0)

    assert len(active_pool) == 0
    assert len(active_pool.unlabelled_pool) == 200

    active_pool.label_based_on_prop([[0.05, 0.45], [0.05, 0.45]], 100)
    assert len(active_pool) == 100

    inds = active_pool._dataset.indices[active_pool._labelled]

    ys = active_pool._dataset.dataset.y[inds]
    ss = active_pool._dataset.dataset.s[inds]

    assert ((ys == 0) & (ss == 0)).sum() / len(active_pool) == 0.05
    assert ((ys == 0) & (ss == 1)).sum() / len(active_pool) == 0.45
    assert ((ys == 1) & (ss == 0)).sum() / len(active_pool) == 0.05
    assert ((ys == 1) & (ss == 1)).sum() / len(active_pool) == 0.45


def test_label_based_on_prop_uneven():
    torch.manual_seed(0)
    dataset = TrainData()
    dataset = Subset(dataset, torch.arange(200))
    active_pool = ActivePool(dataset, 0)

    assert len(active_pool) == 0
    assert len(active_pool.unlabelled_pool) == 200

    active_pool.label_based_on_prop([[0.05, 0.45], [0.05, 0.45]], 97)
    assert len(active_pool) == 97

    inds = active_pool._dataset.indices[active_pool._labelled]

    ys = active_pool._dataset.dataset.y[inds]
    ss = active_pool._dataset.dataset.s[inds]

    assert ((ys == 0) & (ss == 0)).sum() / len(active_pool) == pytest.approx(
        0.05, 0.01, abs=True
    )
    assert ((ys == 0) & (ss == 1)).sum() / len(active_pool) == pytest.approx(
        0.45, 0.01, abs=True
    )
    assert ((ys == 1) & (ss == 0)).sum() / len(active_pool) == pytest.approx(
        0.05, 0.01, abs=True
    )
    assert ((ys == 1) & (ss == 1)).sum() / len(active_pool) == pytest.approx(
        0.45, 0.01, abs=True
    )
