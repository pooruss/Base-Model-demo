import argparse
import pathlib
from abc import ABC
from typing import Iterable, Any
import json

from .dataset import TripleDataset, MaskTripleDataset


class DataLoader(ABC):
    """
    DataLoader is responsible for constructing train, valid and test DataSets from command line arguments.
    """
    train: Iterable[Any]
    valid: Iterable[Any]
    test: Iterable[Any]

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...


class TripleDataLoader(DataLoader):
    train: TripleDataset
    valid: TripleDataset
    test: TripleDataset

    def __init__(self, config: argparse.Namespace):
        self.config = config
        if len(config.data_files) not in [1, 3]:
            raise ValueError("There should be 1 or 3 data files!")
        path = pathlib.Path(config.data_path)
        if len(config.data_files) == 1:
            self.train = TripleDataset(path / config.data_files[0], batch_size=config.train_batch_size)
        else:
            self.train = TripleDataset(path / config.data_files[0], batch_size=config.train_batch_size)
            self.valid = TripleDataset(path / config.data_files[1], batch_size=config.test_batch_size)
            self.test = TripleDataset(path / config.data_files[2], batch_size=config.test_batch_size)
        with open(path / "config.json") as f:
            data_conf = json.load(f)
            config.ent_size = data_conf['ent_size']
            config.rel_size = data_conf['rel_size']

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--data_path', type=str, help="Data path")
        parser.add_argument('--train_batch_size', type=int, default=200, help="Train batch size")
        parser.add_argument('--test_batch_size', type=int, default=200, help="Test batch size")
        parser.add_argument('--data_files', type=str, nargs='+', help="Data filename, e.g. train.npy. If 1 file were"
                                                                      "given, it is treated as the training file, or "
                                                                      "evaluation file during evaluation. If 3 files "
                                                                      "were given, they are treated as training, "
                                                                      "validating and testing data respectively.")
        return parser


class MaskTripleDataLoader(DataLoader):
    train: MaskTripleDataset
    valid: MaskTripleDataset
    test: MaskTripleDataset

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.vocab_path = config.vocab_path
        if len(config.data_files) not in [1, 3]:
            raise ValueError("There should be 1 or 3 data files!")
        path = pathlib.Path(config.data_path)
        if len(config.data_files) == 1:
            self.train = MaskTripleDataset(
                path / config.data_files[0],
                config.vocab_path,
                batch_size=config.train_batch_size)
        else:
            self.train = MaskTripleDataset(
                path / config.data_files[0],
                self.vocab_path,
                batch_size=config.train_batch_size)
            self.valid = MaskTripleDataset(
                path / config.data_files[1],
                self.vocab_path,
                batch_size=config.test_batch_size)
            self.test = MaskTripleDataset(
                path / config.data_files[2],
                self.vocab_path,
                batch_size=config.test_batch_size)
        with open(path / "config.json") as f:
            data_conf = json.load(f)
            config.ent_size = data_conf['ent_size']
            config.rel_size = data_conf['rel_size']

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--data_path', type=str, help="Data path")
        parser.add_argument('--train_batch_size', type=int, default=200, help="Train batch size")
        parser.add_argument('--test_batch_size', type=int, default=200, help="Test batch size")
        parser.add_argument('--data_files', type=str, nargs='+', help="Data filename, e.g. train.npy. If 1 file were"
                                                                      "given, it is treated as the training file, or "
                                                                      "evaluation file during evaluation. If 3 files "
                                                                      "were given, they are treated as training, "
                                                                      "validating and testing data respectively.")
        return parser
