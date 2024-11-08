"""
copy from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
"""
import pdb

import numpy as np
import torch
from model import CustomTrainer
from scipy import stats
from sklearn.metrics import pairwise_distances

from . import StrategyBaseClass


class BADGE(StrategyBaseClass):
    def __init__(self, model, seed, num_samples, classes, **kwargs):
        super().__init__(model, seed, num_samples, **kwargs)
        self._classes = classes

    def __call__(self, trainer: CustomTrainer, loader, **kwargs):
        with torch.no_grad():
            gradEmbedding = trainer.gradient_embeddings(self._classes, self.model, loader)
        gradEmbedding = gradEmbedding.numpy()
        chosen = init_centers(gradEmbedding, self.num_samples)
        return chosen


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    # print("#Samps\tTotal Distance")
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + "\t" + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll
