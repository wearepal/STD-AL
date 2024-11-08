"""
copy from https://github.com/JordanAsh/badge/blob/master/query_strategies/kmeans_sampling.py
"""
import numpy as np
import torch
from model import CustomTrainer
from .utils.cluster import KMeans as _KMeans

from . import StrategyBaseClass


class KMeans(StrategyBaseClass):
    def __init__(self, model, seed, num_samples, fixed_embedding, **kwargs):
        super().__init__(model, seed, num_samples)
        self.fixed_embedding = fixed_embedding
        self.embeddings = None

    def __call__(self, trainer: CustomTrainer, loader, **kwargs):
        if not self.fixed_embedding:
            with torch.no_grad():
                self.embeddings = trainer.embeddings(self.model, loader).numpy()
        else:
            if self.embeddings is None:
                with torch.no_grad():
                    self.embeddings = trainer.embeddings(self.model, loader).numpy()

        cluster_learner = _KMeans(
            d=self.embeddings.shape[1], n_clusters=self.num_samples, seed=self.seed
        )
        cluster_learner.train(self.embeddings)

        cluster_idxs = cluster_learner.predict(self.embeddings)
        centers = cluster_learner.cluster_centers[cluster_idxs]
        dis = (self.embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [
                np.arange(self.embeddings.shape[0])[cluster_idxs == i][
                    dis[cluster_idxs == i].argmin()
                ]
                for i in range(self.num_samples)
            ]
        )
        chosen = q_idxs.tolist()
        if self.fixed_embedding:
            self.embeddings = np.delete(self.embeddings, chosen, 0)
        return chosen
