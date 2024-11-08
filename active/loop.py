import os
from copy import deepcopy

import numpy as np
import wandb
from configs.config import WandbMode
from model import CustomTrainer, ModelBaseClass
from torch.utils.data import DataLoader

from data import CustomSubset

from .data import ActivePool
from .strategy import StrategyBaseClass


class ActiveLoop:
    def __init__(
        self,
        num_steps: int,
        model: ModelBaseClass,
        testset: CustomSubset,
        active_pool: ActivePool,
        strategy: StrategyBaseClass,
        batch_size: int,
        trainer: CustomTrainer,
        num_workers=4,
    ):
        self.active_pool = active_pool
        self.num_steps = num_steps
        self.model = model
        self.testset = testset
        self.batch_size = batch_size
        self.strategy = strategy
        self.trainer = trainer
        self.num_workers = num_workers

    def __call__(self):
        self.on_start()
        for step in range(self.num_steps + 1):
            self.on_loop_start()
            self.training()
            self.test()
            if step < self.num_steps:
                self.query()
            self.on_loop_end()
        self.on_end()

    def training(self):
        loader = DataLoader(
            self.active_pool, self.batch_size, num_workers=self.num_workers
        )
        self.trainer.fit(self.model, loader)
        self.metrics["train/loss"] = np.array(
            [d["loss"] for d in self.model.train_history]
        ).mean()
        self.metrics["train/acc"] = self.model.train_history[-1]["acc"]

    def test(self):
        # test overall
        loader = DataLoader(self.testset, self.batch_size, num_workers=self.num_workers)
        self.trainer.test(self.model, loader)
        self.metrics["test/avg_loss"] = self.model.metrics["test_loss"].compute()
        self.metrics["test/avg_acc"] = self.model.metrics["test_acc"].compute()

        # test y
        for y, subset in self.testset.get_y_subset():
            loader = DataLoader(subset, self.batch_size, num_workers=self.num_workers)
            self.trainer.test(self.model, loader)
            self.metrics[f"test/loss/y={y}"] = self.model.metrics["test_loss"].compute()
            self.metrics[f"test/acc/y={y}"] = self.model.metrics["test_acc"].compute()

        # test s
        acc_s = []
        for s, subset in self.testset.get_s_subset():
            loader = DataLoader(subset, self.batch_size, num_workers=self.num_workers)
            self.trainer.test(self.model, loader)
            self.metrics[f"test/loss/s={s}"] = self.model.metrics["test_loss"].compute()
            acc = self.model.metrics["test_acc"].compute()
            acc_s.append(acc)
            self.metrics[f"test/acc/s={s}"] = acc

        self.metrics["test/robust_acc"] = min(acc_s)

        # test y,s
        for (y, s), subset in self.testset.get_y_s_subset():
            loader = DataLoader(subset, self.batch_size, num_workers=self.num_workers)
            self.trainer.test(self.model, loader)
            self.metrics[f"test/loss/y={y},s={s}"] = self.model.metrics[
                "test_loss"
            ].compute()
            self.metrics[f"test/acc/y={y},s={s}"] = self.model.metrics[
                "test_acc"
            ].compute()

    def query(self):
        loader = DataLoader(
            self.active_pool.unlabelled_pool,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        indices = self.strategy(
            trainer=self.trainer,
            loader=loader,
            w=self.init_weights,
        )
        self.active_pool.label(indices)

    def on_loop_start(self):
        self.metrics = dict()
        self.model.load_state_dict(self.init_weights)
        self.metrics.update(self.active_pool.state)

    def on_start(self):
        self.init_weights = deepcopy(self.model.state_dict())

    def on_end(self):
        try:
            if wandb.config.wandb["mode"].split(".")[1] == WandbMode.disabled.name:
                return
        except KeyError:
            return

        history_path = os.path.join(wandb.run.dir, "labelled_history.npy")
        with open(history_path, "wb") as f:
            np.save(f, np.array(self.active_pool.history))
        wandb.save(history_path)

    def on_loop_end(self):
        self.metrics.update(self.strategy.state_dict())
        wandb.log(self.metrics)
