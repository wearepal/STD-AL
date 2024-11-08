import random

import numpy as np
import torch
import wandb
from active.loop import ActiveLoop
from active.strategy import Strategy
from configs.strategy import Strategy as StrategyEnum
from model import CustomTrainer
from torch.utils.data import Subset

from .test_strategy import NN, ActivePool, CustomSubset, TestData, optim_cfg


def test_loop():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    wandb.init(mode="disabled")

    dataset = TestData("")
    unlabelled = Subset(dataset, torch.arange(0, 200))
    test = CustomSubset(dataset, torch.arange(200, 300))
    active_set = ActivePool(unlabelled, 0)
    active_set.randomly_label(100)

    model = NN(optim_cfg)

    active_loop = ActiveLoop(
        num_steps=2,
        model=model,
        testset=test,
        active_pool=active_set,
        strategy=Strategy({"name": StrategyEnum.Random}, model, 0, 10).get(),
        batch_size=128,
        trainer=CustomTrainer(
            num_epochs=2,
            clamp_grad=False,
        ),
        num_workers=0,
    )

    active_loop()
