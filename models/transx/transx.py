import abc
import argparse
import logging
from typing import Tuple, ClassVar, Type

import torch.optim

import bmkg.data
from ..model import BMKGModel
from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F

from ...data import DataLoader, TripleDataLoader, RandomCorruptSampler, RandomChoiceSampler


class TransX(BMKGModel, ABC):
    def __init__(self, config: argparse.Namespace):
        super(TransX, self).__init__(config)
        self.ent_embed = nn.Embedding(config.ent_size, config.dim, max_norm=1)
        self.rel_embed = nn.Embedding(config.rel_size, config.dim, max_norm=1)
        self.gamma = torch.Tensor([config.gamma]).cuda()
        self.p_norm = config.p_norm

    @abstractmethod
    def scoring_function(self, heads, rels, tails, *args):
        """
        scoring_function defines a scoring function for a TransX-like model.

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :param args: Additional arguments given by dataset.
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """

    def train_step(self, batch):
        pos, neg = self.forward(*batch)
        # we want minimal loss
        # TODO: regularization
        score = F.logsigmoid(self.gamma - pos) + F.logsigmoid(-neg)
        loss = -score.mean()
        # loss = (torch.max(pos - neg, -self.gamma)).mean() + self.gamma
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, pos, neg) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Data
        posh = torch.LongTensor(pos.h).cuda()
        posr = torch.LongTensor(pos.r).cuda()
        post = torch.LongTensor(pos.t).cuda()
        negh = torch.LongTensor(neg.h).cuda()
        negr = torch.LongTensor(neg.r).cuda()
        negt = torch.LongTensor(neg.t).cuda()
        pos_score = self.scoring_function(posh, posr, post)
        neg_score = self.scoring_function(negh, negr, negt)
        return pos_score, neg_score

    def on_train_start(self):
        head_sampler = RandomCorruptSampler(self.train_data, self.config.ent_size, mode='head')
        tail_sampler = RandomCorruptSampler(self.train_data, self.config.ent_size, mode='tail')
        combined = RandomChoiceSampler([head_sampler, tail_sampler])
        self.train_data = combined

    @staticmethod
    def load_data() -> Type[DataLoader]:
        return bmkg.data.TripleDataLoader

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = super().add_args(parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--dim", type=int, default=128, help="The embedding dimension for relations and entities")
        parser.add_argument("--gamma", type=float, default=15.0, help="The gamma for max-margin loss")
        parser.add_argument("--p_norm", type=int, default=2, help="The order of the Norm")
        parser.add_argument("--norm-ord", default=2, help="Ord for norm in scoring function")

        return parser
