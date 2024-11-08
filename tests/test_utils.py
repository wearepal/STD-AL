import numpy as np
import pytest
import torch
from active.strategy.utils.cluster import AgglomerativeClustering, KMeans
from active.strategy.utils.fast_sil import silhouette_samples
from active.strategy.utils.tree import Node, Tree


def test_silhouette_samples():
    n = 1000
    x = np.random.rand(n, 100)
    y = np.random.randint(0, 10, size=n)
    score = silhouette_samples(x, y, cuda=torch.cuda.is_available())
    assert len(score) == n


def test_KMeans():
    n = 1000
    x = np.random.rand(n, 100)
    kmeans = KMeans(d=100, n_clusters=10, seed=0)
    kmeans.train(x)
    assert len(kmeans.predict(x)) == n
    assert kmeans.distances(x).shape == (n, 10)
    assert kmeans.cluster_centers.shape == (10, 100)


def test_AgglomerativeClustering():
    n = 1000
    x = np.random.rand(n, 100)
    agg = AgglomerativeClustering(n_clusters=10)
    agg.train(x)
    assert len(agg.predict(x)) == n


def test_Node():
    n1 = Node(0)
    n1.add_child(Node(1)).add_child(Node(2))

    assert len(n1.children) == 2
    assert 1 in [i.name for i in n1.children]
    assert 2 in [i.name for i in n1.children]
    assert n1.is_root
    with pytest.raises(ValueError):
        n1.add_child(n1)


def test_Tree():
    """
             8
         /       \
        7         \
       / \         \
      5   \         6
     / \   \       / \
    1   2   0     3   4
    # noqa: W605
    """
    children = [[1, 2], [3, 4], [5, 0], [7, 6]]
    root = Tree(children, 5)

    assert isinstance(root, Node)
    assert root.name == 8
    assert root.children[0].name == 7
    assert root.children[1].name == 6

    r = root.children[0]
    assert r.children[0].name == 5
    assert r.children[1].name == 0

    r = r.children[0]
    assert r.children[0].name == 1
    assert r.children[1].name == 2

    r = root.children[1]
    assert r.children[0].name == 3
    assert r.children[1].name == 4

    assert set([i.name for i in root.leafs]) == set(list(range(5)))
    assert set([i.name for i in root.children[0].leafs]) == set([0, 1, 2])
    assert set([i.name for i in root.children[1].leafs]) == set([3, 4])

    r1 = root.children[0]
    assert set([i.name for i in r1.children[0].leafs]) == set([1, 2])
    assert set([i.name for i in r1.children[1].leafs]) == set([0])
    assert r1.children[1].is_leaf

    r1 = r1.children[0]
    assert set([i.name for i in r1.children[0].leafs]) == set([1])
    assert set([i.name for i in r1.children[1].leafs]) == set([2])
    assert r1.children[0].is_leaf
    assert r1.children[1].is_leaf

    r1 = root.children[1]
    assert set([i.name for i in r1.children[0].leafs]) == set([3])
    assert set([i.name for i in r1.children[1].leafs]) == set([4])
    assert r1.children[0].is_leaf
    assert r1.children[1].is_leaf

    assert set([i.name for i in root.subnodes]) == set([7, 5, 6])
    assert set([i.name for i in root.children[0].subnodes]) == set([5])
    assert set([i.name for i in root.children[1].subnodes]) == set([])
