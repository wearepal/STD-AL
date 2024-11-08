import importlib
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import DatasetConfig
from configs.lr_scheduler import LRSchedulerConfig
from configs.optim import OptimConfig
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


class Loss:
    def __init__(self) -> None:
        self.loss = list()

    def reset(self):
        self.loss = list()

    def __call__(self, loss):
        self.loss.append(loss.cpu().detach())

    def compute(self):
        history = torch.tensor(self.loss)
        return history.mean()


class ModelBaseClass(nn.Module):
    def __init__(self, optim_cfg: OptimConfig):
        super().__init__()
        self.metrics = dict()
        self.add_metric("acc", Accuracy)
        self.add_metric("loss", Loss)
        self.criterion = nn.CrossEntropyLoss()
        self.optim_cfg = optim_cfg

    def compute_loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    @property
    def device(self):
        return next(self.parameters()).device

    def add_metric(self, name: str, initializer):
        self.metrics["test_" + name] = initializer()
        self.metrics["train_" + name] = initializer()

    def on_train_start(self) -> None:
        self.train_history = list()

    def on_train_epoch_start(self):
        for name, metric in self.metrics.items():
            if "train" in name:
                metric.reset()

    def predict_step(self, batch):
        batch = batch.to(self.device)
        pred = self.forward(batch)
        return F.softmax(pred, dim=1)

    def training_step(self, batch, weights=None):
        _, (x, y, _) = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        if weights is None:
            loss = self.compute_loss(y_hat, y)
        else:
            self.criterion.reduction = "none"
            weights = weights.to(self.device)
            loss = (self.compute_loss(y_hat, y) * weights).mean()
            self.criterion.reduction = "mean"

        pred = y_hat.cpu().detach()
        if pred.shape[1] == 1:
            pred = pred.squeeze()

        self.metrics["train_acc"](pred, y.cpu())
        self.metrics["train_loss"](loss)
        return loss

    def on_train_epoch_end(self):
        loss = self.metrics["train_loss"].compute()
        acc = self.metrics["train_acc"].compute()
        self.train_history.append({"loss": loss.item(), "acc": acc.item()})

    def configure_optimizers(self):
        if not isinstance(self.optim_cfg, dict):
            kwargs = OmegaConf.to_container(
                self.optim_cfg, enum_to_str=True, resolve=True
            )
        else:
            kwargs = deepcopy(self.optim_cfg)

        name = kwargs["name"]
        del kwargs["name"]

        optimizer = getattr(optim, name)(self.parameters(), **kwargs)
        return optimizer

    def on_test_start(self) -> None:
        for name, metric in self.metrics.items():
            if "test" in name:
                metric.reset()

    def test_step(self, batch):
        _, (x, y, _) = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)

        pred = y_hat.cpu().detach()
        if pred.shape[1] == 1:
            pred = pred.squeeze()
        self.metrics["test_acc"](pred, y.cpu())
        self.metrics["test_loss"](loss)

    @staticmethod
    def compute_gradient(logits, classses, embed):
        batchProbs = F.softmax(logits, dim=1)
        maxInds = torch.argmax(batchProbs, 1)
        g = []
        for c in classses:
            g.append(embed * ((maxInds == c).float() - batchProbs[:, c]).unsqueeze(1))
        return torch.hstack(g)


class CustomTrainer:
    def __init__(
        self,
        num_epochs=None,
        clamp_grad=None,
        lr_scheduler_config: Optional[LRSchedulerConfig] = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.clamp_grad = clamp_grad
        self.lr_scheduler_config = lr_scheduler_config

    def _setup_lr_scheduler(self, optimizer):
        if self.lr_scheduler_config:
            if self.lr_scheduler_config["name"] is None:
                return None
            if not isinstance(self.lr_scheduler_config, dict):
                kwargs = OmegaConf.to_container(
                    self.lr_scheduler_config, enum_to_str=True, resolve=True
                )
            else:
                kwargs = deepcopy(self.lr_scheduler_config)

            del kwargs["name"]

            return eval(f"optim.lr_scheduler.{self.lr_scheduler_config['name']}")(
                optimizer,
                **kwargs,
            )
        return None

    def fit(
        self,
        model: ModelBaseClass,
        loader: DataLoader,
        weights=None,
    ):
        model.on_train_start()
        optimizer = model.configure_optimizers()
        lr_scheduler = self._setup_lr_scheduler(optimizer)
        for _ in range(self.num_epochs):
            model.on_train_epoch_start()
            for batch in loader:
                if weights is not None:
                    loss = model.training_step(batch, weights[batch[0]])
                else:
                    loss = model.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                if self.clamp_grad:
                    for p in filter(lambda p: p.grad is not None, model.parameters()):
                        p.grad.data.clamp_(min=self.clamp_grad[0], max=self.clamp_grad[1])
                optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            model.on_train_epoch_end()
        return {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
        }

    def test(self, model: ModelBaseClass, loader: DataLoader):
        model.on_test_start()
        for batch in loader:
            model.test_step(batch)

    def predict(self, model: ModelBaseClass, loader: DataLoader):
        predictions = []
        for batch in loader:
            pred = model.predict_step(batch).detach().cpu()
            predictions.append(pred)
        return torch.vstack(predictions)

    def gradient_embeddings(
        self,
        classses,
        model: ModelBaseClass,
        dataloader: DataLoader,
        return_embedding=False,
    ):
        gradient = []
        if return_embedding:
            feature_embeddings = []
        for x in dataloader:
            if isinstance(x, list) or isinstance(x, tuple):
                x = x[0]
            x = x.to(model.device)
            embed, logits = model.with_embedding(x)
            embed = embed.detach().cpu()
            logits = logits.detach().cpu()
            if return_embedding:
                feature_embeddings.append(embed)
            g = model.compute_gradient(logits, classses, embed)
            gradient.append(g)
        gradient = torch.vstack(gradient)
        if return_embedding:
            return torch.vstack(feature_embeddings), gradient
        return gradient

    def embeddings(self, model: ModelBaseClass, dataloader: DataLoader):
        embeddings = []
        for x in dataloader:
            if isinstance(x, list) or isinstance(x, tuple):
                x = x[0]
            x = x.to(model.device)
            embed, _ = model.with_embedding(x)
            embed = embed.detach().cpu()
            embeddings.append(embed)
        return torch.vstack(embeddings)


class Model:
    def __init__(self, data_cfg: DatasetConfig, optim_cfg: OptimConfig):
        self._model = importlib.import_module(f"model.{data_cfg.name}").get_model(
            data_cfg, optim_cfg
        )

    def get(self):
        return self._model
