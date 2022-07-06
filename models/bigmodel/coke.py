import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
from transformers import BertConfig, BertModel


class CoKE(nn.Module):
    """CoKE: Contextualized Knowledge Graph Embedding."""

    # TODO: soft label, attn_mask scale, activation function.
    def __init__(self, config: argparse.Namespace):
        super().__init__()
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
        self.mask_id = config['mask_id']
        self.config = BertConfig.from_pretrained(
            pretrained_model_name_or_path='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/')
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/')
        self.config = self.bert.config
        self.dropout_layer = nn.Dropout(self.dropout)
        self.mlm_ffn = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size, eps=1e-12),
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.vocab_size, bias=True)
        )
        self.init_parameters()
        self.tokenizer = None

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                # m.linear1.bias.data.zero_()
                m.linear1.weight.data.normal_(mean=0.0, std=self.initializer_range)
                # m.linear2.bias.data.zero_()
                m.linear2.weight.data.normal_(mean=0.0, std=self.initializer_range)
                m.norm1.bias.data.zero_()
                m.norm1.weight.data.fill_(1.0)
                m.norm2.bias.data.zero_()
                m.norm2.weight.data.fill_(1.0)
                m.self_attn.out_proj.weight.data.normal_(mean=0.0, std=self.initializer_range)
                # m.self_attn.out_proj.bias.data.zero_()
            # elif isinstance(m, nn.MultiheadAttention):
            #     m.out_proj.weight.data.normal_(mean=0.0, std=self.initializer_range)
            #     m.out_proj.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                # nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)

    def load_pretrained_embedding(self, path):
        # self.load_state_dict()
        state_dict = torch.load(os.path.join(path))
        for key, value in state_dict.items():
            if key == 'ent_embeddings.weight':
                entity_embed = value
            if key == 'rel_embeddings.weight':
                rel_embed = value
        coke_embed = torch.cat((entity_embed, rel_embed), dim=0)
        self.embedding_layer["word_embedding"].weight = nn.Parameter(coke_embed)
        self.linear2.weight = self.embedding_layer["word_embedding"].weight

    def forward(self, input_map):
        src_ids = input_map['src_ids'].squeeze()
        input_mask = input_map['input_mask'].squeeze()
        position_ids = input_map['position_ids'].squeeze()
        segment_ids = input_map['segment_ids'].squeeze()
        # input_mask = input_map['input_mask'].squeeze(dim=1)
        # emb_out = self.embedding_layer["word_embedding"](src_ids) + \
        #           self.embedding_layer["position_embedding"](position_ids)
        # emb_out = self.embedding_layer["layer_norm"](emb_out)
        # emb_out = self.embedding_layer["dropout"](emb_out)
        # emb_out = emb_out.permute(1, 0, 2)  # -> seq_len x B x E

        # with torch.no_grad():
        # self_attn_mask = torch.bmm(input_mask, input_mask.permute(0, 2, 1)) * -10000
        # attn_mask = torch.stack(tensors=[self_attn_mask] * self.n_head, dim=1).squeeze()
        # attn_mask = attn_mask.reshape(shape=[src_ids.size(0) * self.n_head, -1, self.max_seq_len])
        # attn_mask.requires_grad = False

        # backbone
        # enc_out = self.transformer_block["transformer_encoder"](emb_out)
        # enc_out = self.bert(token_type_ids=segment_ids, inputs_embeds=emb_out, encoder_attention_mask=input_mask)
        enc_out = self.bert(
            input_ids=src_ids,
            position_ids=position_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask
        )

        # pooled_output = enc_out.pooler_output
        # pooled_output = self.dropout_layer(pooled_output)
        # pooled_output = self.classifier(pooled_output)

        last_hidden_state = enc_out.last_hidden_state
        # method 1
        # enc_out = enc_out.reshape(shape=[-1, self.emb_size])
        # enc_out = torch.index_select(input=enc_out, dim=0, index=mask_pos)

        # # method 2
        # enc_out = enc_out.transpose(0, 1)
        # enc_out = enc_out[torch.arange(mask_pos.shape[0]), mask_pos, :]

        # # method 3
        # enc_out = enc_out.transpose(0, 1)
        # enc_out = enc_out[src_ids == 50264]
        # # method 4
        # mlm_last_hidden_state = last_hidden_state.view(-1, self.config.hidden_size)[mlm_mask_pos.view(-1)]
        # mlm_last_hidden_state = self.mlm_ffn(mlm_last_hidden_state.view(-1, self.config.hidden_size))
        # mem_last_hidden_state = last_hidden_state.view(-1, self.config.hidden_size)[mem_mask_pos.view(-1)]
        # mem_last_hidden_state = self.mem_ffn(mem_last_hidden_state.view(-1, self.config.hidden_size))
        last_hidden_state = self.mlm_ffn(last_hidden_state)
        # mlm_last_hidden_state = last_hidden_state.view(-1, self.config.vocab_size)[mlm_mask_pos.view(-1)]
        # mem_last_hidden_state = last_hidden_state.view(-1, self.config.vocab_size)[mem_mask_pos.view(-1)]
        output_map = {
            'logits': last_hidden_state
            # 'pooled_output': pooled_output
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
