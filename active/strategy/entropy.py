"""
copy from https://github.com/JordanAsh/badge/blob/master/query_strategies/margin_sampling.py
"""
import torch
from model import CustomTrainer

from . import StrategyBaseClass, to_multiclass_prob


class Entropy(StrategyBaseClass):
    def __call__(self, trainer: CustomTrainer, loader, **kwargs):
        probs = trainer.predict(self.model, loader)
        probs = to_multiclass_prob(probs)
        log_probs = torch.log(probs)
        U = (probs * log_probs).sum(1)
        return U.sort()[1].numpy()[: self.num_samples].tolist()
