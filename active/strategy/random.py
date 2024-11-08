from . import StrategyBaseClass


class Random(StrategyBaseClass):
    def __call__(self, loader, **kwargs):
        indices = self.rnd.permutation(len(loader.dataset))
        return indices[: self.num_samples].tolist()
