import argparse
from email import header

import torch
import torch.nn as nn
#from . import BMKGModel
import torch.nn.functional as F


# import  fairseq.models.transformer.TransformerEncoder as TransformerEncoder
from transformers import RobertaConfig, RobertaModel

class CoKE_Roberta(nn.Module):
    """CoKE: Contextualized Knowledge Graph Embedding."""

    # TODO: soft label, attn_mask scale, activation function.
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)
        self.max_seq_len = config['max_seq_len']
        self.emb_size = config['hidden_size']
        self.n_layer = config['num_hidden_layers']
        self.n_head = config['num_attention_heads']
        self.voc_size = config['vocab_size']
        self.n_relation = config['num_relations']
        self.max_position_seq_len = config['max_position_embeddings']
        self.dropout = config['dropout']
        self.attention_dropout = config['attention_dropout']
        self.intermediate_size = config['intermediate_size']
        self.weight_sharing = config['weight_sharing']
        self.initializer_range = config['initializer_range']

        config = RobertaConfig()
        config.num_attention_heads = self.n_head
        config.intermediate_size = self.intermediate_size
        config.hidden_size =  self.emb_size
        config.max_position_embeddings = self.max_position_seq_len
        config.vocab_size = self.voc_size
        self.RobertaModel = RobertaModel(config)

        # embedding layer
        self.embedding_layer = nn.ModuleDict({
            'word_embedding': nn.Embedding(num_embeddings=self.voc_size, embedding_dim=self.emb_size),
            'position_embedding': nn.Embedding(num_embeddings=self.max_position_seq_len, embedding_dim=self.emb_size),
            'layer_norm': nn.LayerNorm(self.emb_size, eps=1e-12),
            'dropout': nn.Dropout(p=self.dropout)
        })

        # # transformer block
        # self.transformer_block = nn.ModuleDict({
        #     'transformer_encoder':
        #         nn.TransformerEncoder(
        #             nn.TransformerEncoderLayer(
        #                 d_model=self.emb_size,
        #                 nhead=self.n_head,
        #                 dim_feedforward=self.intermediate_size,
        #                 layer_norm_eps=1e-5,
        #                 dropout=self.attention_dropout,
        #                 activation='gelu'),
        #             num_layers=12)
        # })


        # self.LSTM = nn.LSTM(input_size=self.emb_size,
        #             hidden_size=self.emb_size//2,
        #             num_layers = 1,
        #             batch_first=True,
        #             bidirectional=True)

        # classification
        self.linear1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear2 = nn.Linear(in_features=self.emb_size, out_features=self.voc_size)
        self.linear2.weight = self.embedding_layer["word_embedding"].weight
        self.fc = nn.Sequential(
                    self.linear1,
                    nn.GELU(),
                    nn.LayerNorm(self.emb_size, eps=1e-12),
                    self.linear2)
        
        self.init_parameters()

        # def init_parameters(self):
        #     for m in self.modules():
        #         # if isinstance(m, nn.Linear):
        #         #     nn.init.normal_(m.weight, 0, self.initializer_range)
        #         #     nn.init.constant_(m.bias, 0)

        #         if isinstance(m, (nn.Linear, nn.Embedding)):
        #             # cf https://github.com/pytorch/pytorch/pull/5617
        #             m.weight.data.normal_(mean=0.0, std=self.initializer_range)
        #         elif isinstance(m, nn.LayerNorm):
        #             m.bias.data.zero_()
        #             m.weight.data.fill_(1.0)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             m.bias.data.zero_()

        # def init_parameters(self):
        #     nn.init.xavier_uniform_(self.embedding_layer['word_embedding'].weight.data)
        #     nn.init.xavier_uniform_(self.embedding_layer['position_embedding'].weight.data)
        #     nn.init.xavier_uniform_(self.classification_head['linear1'].weight.data)
        #     for m in self.transformer_block.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, 0, self.initializer_range)
        #             nn.init.constant_(m.bias, 0)
    def init_parameters(self):
        r"""Initiate parameters in the transformer model."""

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         # nn.init.xavier_uniform_(p)
        #         nn.init.normal_(p, 0, self.initializer_range)
        # nn.init.xavier_uniform_(self.embedding_layer['word_embedding'].weight.data)
        # nn.init.xavier_uniform_(self.embedding_layer['position_embedding'].weight.data)
        nn.init.normal_(self.linear1.weight.data,0,self.initializer_range)
        nn.init.normal_(self.embedding_layer['position_embedding'].weight.data,0,self.initializer_range)
        nn.init.normal_(self.embedding_layer['word_embedding'].weight.data,0,self.initializer_range)

    def forward(self, input_map):
        src_ids = input_map['src_ids'].squeeze()
        position_ids = input_map['position_ids'].squeeze()
        mask_pos = input_map['mask_pos'].squeeze()
        input_mask = input_map['input_mask'].squeeze(dim=1)
        emb_out = self.embedding_layer["word_embedding"](src_ids)  +  self.embedding_layer["position_embedding"](position_ids)
        emb_out = self.embedding_layer["layer_norm"](emb_out)
        # enc_out,_ = self.LSTM(emb_out)
        enc_out = self.RobertaModel(inputs_embeds=emb_out )[0]
        # from IPython import embed; embed(header='First time')
        enc_out = enc_out[src_ids==99]
        logits = self.fc(enc_out)
        # logits[:,99]=0
        output_map = {
            "logits":logits
        }
        return output_map




    def configure_optimizers(self):
        pass

    def train_step(self, train_data_loader, **kwargs):

        pass

    def load_data(self):
        pass

    def compute_loss(self):
        pass

 

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
