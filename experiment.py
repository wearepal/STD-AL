import os
import random

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from active.data import ActivePool, log_labelled, log_unlabelled
from active.loop import ActiveLoop
from active.strategy import Strategy
from configs import config
from configs import lr_scheduler as lr_scheduler_config  # type: ignore # noqa
from configs import optim as optim_config  # type: ignore # noqa
from configs import strategy as strategy_config  # type: ignore # noqa
from configs.lr_scheduler import LRScheduler as LRSchedulerEnum
from configs.optim import Optim as OptimEnum
from configs.strategy import Strategy as StrategyEnum
from data import Dataset
from model import CustomTrainer, Model

cs = ConfigStore.instance()
cs.store(name="config_schema", node=config.Config)
cs.store(group="schema/data", package="data", name="cmnist", node=config.CMNISTConfig)
cs.store(group="schema/data", package="data", name="celeba", node=config.CelebAConfig)
cs.store(
    group="schema/data", package="data", name="synthetic", node=config.SyntheticConfig
)

for o in OptimEnum:
    cs.store(
        group="schema/optim",
        package="optim",
        name=str(o),
        node=eval(f"optim_config.{o}"),
    )

for s in StrategyEnum:
    cs.store(
        group="schema/strategy",
        package="strategy",
        name=str(s),
        node=eval(f"strategy_config.{s}"),
    )

for s in LRSchedulerEnum:
    cs.store(
        group="schema/lr_scheduler",
        package="lr_scheduler",
        name=str(s),
        node=eval(f"lr_scheduler_config.{s}"),
    )


os.environ["WANDB_START_METHOD"] = "forkserver"


@hydra.main(config_path="configs", config_name="cfg")
def main(cfg: config.Config):
    import torch

    torch.manual_seed(cfg.exp.seed)
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        reinit=True,
        allow_val_change=True,
        mode=cfg.wandb.mode.name,
        group=f"{cfg.strategy.name}_{cfg.data.name}_{cfg.exp.seed}",
        job_type="AL",
        config=OmegaConf.to_container(cfg),
    )

    datapath = "/srv/galene0/shared/data/" if torch.cuda.is_available() else "/tmp"

    model = Model(cfg.data, cfg.optim).get()

    if torch.cuda.is_available():
        model.cuda()

    classes, train, test = Dataset(datapath, cfg.exp.dist, cfg.data, cfg.exp.seed).get()
    active_set = ActivePool(train, cfg.exp.seed)
    if cfg.exp.dist.initial:
        active_set.label_based_on_prop(cfg.exp.dist.initial, cfg.exp.initial)
    else:
        active_set.randomly_label(cfg.exp.initial)

    log_labelled(active_set, "initial")
    log_unlabelled(active_set, "unlabelled")

    strategy = Strategy(
        cfg.strategy,
        model,
        cfg.exp.seed,
        cfg.exp.n_to_label,
        fixed_embedding=model.fixed_embedding,
        active_set=active_set,
        classes=classes,
    ).get()

    active_loop = ActiveLoop(
        num_steps=cfg.exp.num_steps,
        model=model,
        testset=test,
        active_pool=active_set,
        strategy=strategy,
        batch_size=cfg.exp.batch_size,
        trainer=CustomTrainer(
            num_epochs=cfg.exp.num_epochs,
            clamp_grad=cfg.exp.clamp_grad,
            lr_scheduler_config=cfg.lr_scheduler,
        ),
    )

    active_loop()


if __name__ == "__main__":
    main()
