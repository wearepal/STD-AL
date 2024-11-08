"""
copy from https://github.com/JordanAsh/badge/blob/master/query_strategies/margin_sampling.py
"""

from model import CustomTrainer

from . import StrategyBaseClass, to_multiclass_prob


class Margin(StrategyBaseClass):
    def __call__(self, trainer: CustomTrainer, loader, **kwargs):
        self._state = dict()
        probs = trainer.predict(self.model, loader)
        probs = to_multiclass_prob(probs)
        probs_sorted, _ = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        self._state["strategy/margin_score"] = U.numpy()
        return U.sort()[1].numpy()[: self.num_samples].tolist()

    def state_dict(self):
        return self._state
