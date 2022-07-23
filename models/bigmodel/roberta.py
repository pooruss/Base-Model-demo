import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
from transformers import RobertaConfig, RobertaModel


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, selected_hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(selected_hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RobertaBase(nn.Module):
    # TODO: soft label, attn_mask scale, activation function.
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.initializer_range = config['initializer_range']
        self.mask_id = config['mask_id']
        self.pretrained_model_path = config['pretrained_path']
        self.config = RobertaConfig.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_path)
        self.model = RobertaModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_path)
        self.lm_head = RobertaLMHead(self.config)
        self.pooler = RobertaPooler(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
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
        if "mem_mask_pos" in input_map:
            pos_neg_seq = input_map['pos_neg_seq']
            mem_mask_pos = input_map['mem_mask_pos']
            neg_nums = input_map['neg_nums']
            bz = input_map['batch_size']
            pos_seq = pos_neg_seq[:, 0, :].reshape(bz, -1)
            neg_seq = pos_neg_seq[:, 1:, :].reshape(bz * neg_nums, -1)
            src_ids = torch.cat((src_ids, pos_seq, neg_seq), dim=0)
            input_mask = input_mask.repeat(2 + neg_nums, 1)

        enc_out = self.model(
            input_ids=src_ids,
            attention_mask=input_mask
        )
        last_hidden_state = enc_out.last_hidden_state

        if "mem_mask_pos" in input_map:
            mem_mask_pos = mem_mask_pos.repeat(neg_nums, 1)
            neg_hidden_state = last_hidden_state[bz + bz:, :, :][mem_mask_pos]
            pos_hidden_state = last_hidden_state[bz: bz + bz, :, :]
            pos_hidden_state = pos_hidden_state.repeat(neg_nums, 1, 1)[mem_mask_pos]
            total_hidden_state = torch.cat((neg_hidden_state, pos_hidden_state), dim=0)
            total_hidden_state = self.pooler(total_hidden_state)
            total_logits = self.classifier(total_hidden_state)
            # print(total_logits)
            total_logits = F.sigmoid(total_logits)
            # print(total_logits)
            logits = self.lm_head(last_hidden_state[:bz])
            output_map = {
                'logits': logits,
                'neg_pos_logits': total_logits
            }
        else:
            logits = self.lm_head(last_hidden_state)
            output_map = {
                'logits': logits
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
