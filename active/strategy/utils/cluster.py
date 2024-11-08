from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from sklearn.cluster import KMeans as _KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             euclidean_distances)

from .fast_sil import silhouette_samples
from .tree import Tree


class ClusteringMetrics:
    def __init__(self, x, pred_y, true_y=None) -> None:
        self.x = x
        self.pred_y = pred_y
        self.true_y = true_y

    def _get_cluster_sils(self, compute_sil=True):
        """
        copy from
        https://github.com/HazyResearch/hidden-stratification/blob/master/stratification/cluster/models/cluster.py
        """
        unique_preds = sorted(np.unique(self.pred_y))
        SIL_samples = (
            silhouette_samples(self.x, self.pred_y) if compute_sil else np.zeros(len(self.x))
        )
        SILs_by_cluster = {
            int(label): float(np.mean(SIL_samples[self.pred_y == label])) for label in unique_preds
        }
        SIL_global = float(np.mean(SIL_samples))
        return SILs_by_cluster, SIL_global

    def _get_ami(self):
        return adjusted_mutual_info_score(self.true_y, self.pred_y)

    def _get_adjusted_rand_score(self):
        return adjusted_rand_score(self.true_y, self.pred_y)

    def compute(self):
        sil_c, sil_g = self._get_cluster_sils()
        if self.true_y is not None:
            ami = self._get_ami()
            rand = self._get_adjusted_rand_score()
        else:
            ami = rand = None
        return {"sil_c": sil_c, "sil_g": sil_g, "ami": ami, "rand": rand}


class Clustering(ABC):
    @abstractmethod
    def train(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def distances(self, x):
        pass

    def train_predict(self, x):
        self.train(x)
        return self.predict(x)


class SKKmeans(Clustering):
    def __init__(self, n_clusters, seed=0, *args, **kwargs) -> None:
        super().__init__()
        self.kmeans = _KMeans(n_clusters=n_clusters, random_state=seed)

    def train(self, x):
        self.kmeans.fit(x)

    def predict(self, x):
        return self.kmeans.predict(x)

    def distances(self, x):
        return self.kmeans.transform(x)

    @property
    def cluster_centers(self):
        return self.kmeans.cluster_centers_


class FaissKmeans(Clustering):
    def __init__(self, d, n_clusters, seed=0, *args, **kwargs) -> None:
        super().__init__()
        from faiss import Kmeans  # type: ignore # noqa

        self.kmeans = Kmeans(d, n_clusters, seed=seed)

    def train(self, x):
        x = self._cast(x)
        self.kmeans.train(x)

    def predict(self, x):
        x = self._cast(x)
        return self.kmeans.assign(x)[1]

    def distances(self, x):
        x = self._cast(x)
        return euclidean_distances(x, self.kmeans.centroids)

    @property
    def cluster_centers(self):
        return self.kmeans.centroids

    @staticmethod
    def _cast(x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = x.astype(np.float32)
        return x


class KMeans(FaissKmeans if torch.cuda.is_available() else SKKmeans):
    pass


class AgglomerativeClustering(Clustering):
    def __init__(self, n_clusters, *args, **kwargs) -> None:
        super().__init__()
        self.agg = _AgglomerativeClustering(n_clusters, **kwargs)

    def train(self, x):
        self._predict = self.agg.fit_predict(x)
        self._n = len(x)

    def predict(self, _):
        return self._predict

    def distances(self, _):
        raise NotImplementedError

    @property
    def tree(self) -> Tree:
        return Tree(self.agg.children_, self._n)
