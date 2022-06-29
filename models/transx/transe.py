import argparse
import torch

from .transx import TransX


class TransE(TransX):
    def __init__(self, config: argparse.Namespace):
        super(TransE, self).__init__(config)

    def scoring_function(self, heads, rels, tails, *_):
        score = self.ent_embed(heads) + self.rel_embed(rels) - self.ent_embed(tails)
        score = torch.norm(score, p=self.p_norm, dim=-1)
        return score
