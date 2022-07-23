import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
import bmtrain as bmt
from model_center.layer import Linear, LayerNorm
from model_center.model import Roberta, RobertaConfig


class RoertaLMHead(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, norm_eps):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.act_fn = torch.nn.functional.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps)
        self.decoder = Linear(dim_model, vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class RobertaBaseBMT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # config = BertConfig.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.initializer_range = config['initializer_range']
        self.mask_id = config['mask_id']
        self.pretrained_model_path = config['pretrained_path']
        self.vocab_size = config['vocab_size']
        self.config = RobertaConfig.from_pretrained("roberta-base")
        # self.bert = Bert.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.roberta = Roberta.from_pretrained("roberta-base")
        self.lm_head = RoertaLMHead(dim_model=self.config.dim_model,vocab_size=self.vocab_size, norm_eps=1e-5)
        self.mask_id = config['mask_id']

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

