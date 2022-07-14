import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
from transformers import BertConfig, BertModel, BertLMHeadModel


class BertBase(nn.Module):
    """CoKE: Contextualized Knowledge Graph Embedding."""

    # TODO: soft label, attn_mask scale, activation function.
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.initializer_range = config['initializer_range']
        self.mask_id = config['mask_id']
        self.pretrained_model_path = config['pretrained_path']
        self.config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_path)

        self.linear1 = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size)
        self.linear2 = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.config.vocab_size))
        self.linear2.bias = self.bias

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/')

        if config["do_train"]:
            self.init_parameters()
            self.linear2.weight = nn.Parameter(self.bert.embeddings.word_embeddings.weight)
            # print("embedding weight sharing")
        self.mlm_ffn = nn.Sequential(
            self.linear1,
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size, eps=1e-12),
            nn.Dropout(0.1),
            self.linear2
        )
        # self.contrastive_ffn = nn.Sequential(
        #     self.linear1,
        #     self.act,
        #     self.ln,
        #     self.linear3
        # )
        self.tokenizer = None

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(self, input_map):
        src_ids = input_map['src_ids']
        input_mask = input_map['input_mask']
        position_ids = input_map['position_ids']
        segment_ids = input_map['segment_ids']
        enc_out = self.bert(
            input_ids=src_ids,
            position_ids=position_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask
        )

        last_hidden_state = enc_out.last_hidden_state
        last_hidden_state = self.mlm_ffn(last_hidden_state)
        output_map = {
            'logits': last_hidden_state
        }
        return output_map

    def on_train_start(self):
        head_sampler = RandomCorruptMaskSampler(self.train_data, self.config.ent_size, mode='head')
        tail_sampler = RandomCorruptMaskSampler(self.train_data, self.config.ent_size, mode='tail')
        combined = RandomChoiceMaskSampler([head_sampler, tail_sampler])
        self.train_data = combined

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_step(self, batch):
        input_map = dict()
        input_map['src_ids'] = batch[0]
        input_map['position_ids'] = batch[1]
        input_map['mask_pos'] = batch[2]
        input_map['input_mask'] = batch[3]
        labels = batch[4].squeeze()

        output_map = self.forward(input_map)

        logits = output_map['logits']
        loss = F.cross_entropy(logits, labels, label_smoothing=0.8)
        self.log("train/loss", loss)
        return loss

    @staticmethod
    def load_data():
        return MaskTripleDataLoader

    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--max_seq_len', type=int, default=3, help="Number of tokens of the longest sequence.")
        parser.add_argument('--pad_id', type=int, default=-100, help="<pad> id in vocab")
        parser.add_argument('--hidden_size', type=int, default=256, help="CoKE model config, default 256")
        parser.add_argument('--num_hidden_layers', type=int, default=6, help="CoKE model config, default 6")
        parser.add_argument('--num_attention_heads', type=int, default=4, help="CoKE model config, default 4")
        parser.add_argument('--vocab_size', type=int, default=16396, help="CoKE model config")
        parser.add_argument('--num_relations', type=int, default=0, help="CoKE model config")
        parser.add_argument('--max_position_embeddings', type=int, default=10, help="max position embeddings")
        parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
        parser.add_argument('--attention_dropout', type=float, default=0.1, help="attention dropout")
        parser.add_argument('--intermediate_size', type=float, default=512, help="intermediate size")

        return parser
