import abc
import argparse
import logging
from abc import abstractmethod
from typing import Union, Type, Iterable

import torch
import tqdm
import wandb

from ..data import DataLoader


class BMKGModel(abc.ABC, torch.nn.Module):
    step = 0
    epoch = 0
    data_loader: DataLoader
    train_data: Iterable
    valid_data: Iterable
    test_data: Iterable
    pbar: tqdm.tqdm

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.lr = self.config.lr
        self.max_epoch = config.max_epoch
        wandb.init(
            project="BMKG",
            tags=[config.model],
            config=config
        )
        # TODO: INITIALIZE LOGGER

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    def valid_step(self, *args, **kwargs):
        self.train_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        self.train_step(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def load_data() -> Type[DataLoader]:
        pass

    def on_train_start(self) -> None:
        """
        on_train_start hook will be called before train starts.

        by default, we do nothing.
        :return:
        """
        pass

    def do_train(self, data_loader: DataLoader):
        self.train_data = data_loader.train
        self.on_train_start()
        self.train()
        torch.set_grad_enabled(True)
        optim = self.configure_optimizers()
        self.pbar = tqdm.tqdm(total=self.max_epoch * len(self.train_data))
        for i in range(self.max_epoch):
            for data in self.train_data:
                self.step += 1
                loss = self.train_step(data)
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.pbar.update(1)

    def do_valid(self):
        raise NotImplementedError
        # TODO:
        # Call load_data
        # etc.

    def do_test(self):
        raise NotImplementedError
        # TODO:
        # Call load_data
        # etc.

    def log(self, key: str, value: Union[int, float, torch.TensorType]):
        wandb.log({
            key: value
        }, step=self.step)
        # raise NotImplementedError

    def log_hyperparameters(self):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.5, help="Learning rate")
        parser.add_argument('--max_epoch', type=int, default=1, help="How many epochs to run")
        parser.add_argument('--logger', choices=['wandb', 'none'], default='wandb', help="Which logger to use")
        return parser
