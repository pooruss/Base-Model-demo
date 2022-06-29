import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..model import BMKGModel
# from ..transx import transe
# from ..semantic import distmult
# from bmkg.criterion import MLM_loss, KE_loss
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel, RobertaTokenizer, RobertaConfig


# from fairseq.models.roberta import RobertaEncoder

class KEPLER(nn.Module):
    """Model for Knowledge Embedding and Masked Language Modeling"""

    # TODO: activation function, total loss, model zoo.
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        # Default base model: roberta-base
        # self.tokenizer = AutoTokenizer.from_pretrained("./roberta-base/")
        # self.encoder = AutoModel.from_pretrained("roberta-base")
        # self.roberta_config = AutoConfig.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_root)
        self.roberta_config = RobertaConfig.from_pretrained(self.config.model_root)
        self.encoder = RobertaModel.from_pretrained(self.config.model_root)
        # self.encoder = RobertaEncoder(self.config)
        self.ke_head = KEPLERKnowledgeEmbeddingHead(self.config, self.roberta_config.hidden_size)
        self.classification_head = KEPLERClassificationHead(
            input_dim=self.roberta_config.hidden_size,
            inner_dim=self.roberta_config.hidden_size,
            num_classes=self.config.num_classes,
            pooler_dropout=self.config.pooler_dropout
        )
        self.lm_head = KEPLERLMHead(
            embed_dim=self.roberta_config.hidden_size,
            output_dim=self.roberta_config.vocab_size,
            weight=self.encoder.embeddings.word_embeddings.weight,
        )

    def forward(self, input_map):
        """
        :param input_map: a dict contains preprocessed mlm input and ke input
        :return: an output map which is a dict contains losses and features
        """
        mlm_inputs = input_map['mlm_inputs']
        outputs = self.encoder(**mlm_inputs)
        last_hidden_states = outputs[0]
        cls_out = outputs[1]
        logits = self.lm_head(last_hidden_states)
        # logits = logits.transpose(0, 1)
        if input_map['classification_head']:
            cls_out = self.classification_head(last_hidden_states)
        output_map = {
            'logits': logits,
            'mlm_last_hidden_states': last_hidden_states,
            'mlm_cls_out': cls_out
        }
        return output_map

    def ke_score(self, src_tokens, relations):
        heads, tails, nheads, ntails, heads_r, tails_r, relation_desc = src_tokens
        size = heads.size(0)
        head_embs = self.encoder(heads)[0]
        tail_embs = self.encoder(tails)[0]
        nhead_embs = self.encoder(nheads)[0]
        ntail_embs = self.encoder(ntails)[0]
        head_embs_r = self.encoder(heads_r)[0]
        tail_embs_r = self.encoder(tails_r)[0]
        if relation_desc is not None:
            relation_desc_emb, _ = self.encoder(relation_desc)
        else:
            relation_desc_emb = None
        pos_score, neg_score = self.ke_head(head_embs, tail_embs, nhead_embs, ntail_embs, head_embs_r,
                                            tail_embs_r, relations, relation_desc_emb=relation_desc_emb)

        return pos_score, neg_score, size

    def base_model_zoo(self):
        model_zoo = {
            'roberta_base': 'roberta-base',
            'roberta_large': 'roberta-large'
        }
        assert self.args.base_model not in model_zoo, \
            "Base model {} for KEPLER not supported.".format(self.args.base_model)
        return model_zoo

    def register_base_model(self, name):
        """Register a base model for KE and MLM."""
        model_zoo = self.base_model_zoo()
        self.tokenizer = AutoTokenizer.from_pretrained(model_zoo[name])
        self.encoder = AutoModel.from_pretrained(model_zoo[name])
        self.roberta_config = AutoConfig.from_pretrained(model_zoo[name])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_step(self, batch):
        input_map = dict()
        input_map['mlm_inputs'] = batch['mlm_inputs']
        input_map['classification_head'] = batch['classification_head']
        ke_inputs = batch['ke_inputs']
        mlm_targets = batch['mlm_targets']
        ke_targets = batch['ke_targets']

        output_map = self.forward(input_map)
        logits = output_map['logits']

        # MLM Loss
        mlm_loss = MLM_loss(logits=logits, targets=mlm_targets, padding_idx=self.config.padding_idx)
        # KE Loss
        pos_score, neg_score, sample_size = self.ke_score(
            src_tokens=(
                ke_inputs["heads"],
                ke_inputs["tails"],
                ke_inputs["nheads"],
                ke_inputs["ntails"],
                ke_inputs["heads_r"],
                ke_inputs["tails_r"],
                ke_inputs['relation_desc'] if 'relation_desc' in ke_inputs else None),
            relations=ke_targets)
        ke_loss = KE_loss(pos_score=pos_score, neg_score=neg_score)
        print(mlm_loss)
        print(ke_loss)

        loss = mlm_loss + ke_loss
        # self.log("train/loss", loss)
        return loss

    def load_data(self):
        return

    # TODO: add args according to Config
    @staticmethod
    def add_args(parser):
        pass


class KEPLERKnowledgeEmbeddingHead(nn.Module):
    """Head for knowledge embedding pretraining tasks."""

    def __init__(self, args, embed_dim, gamma=0, nrelation=0):
        super().__init__()
        if gamma == 0:
            gamma = args.gamma
        if nrelation == 0:
            nrelation = args.nrelation

        self.args = args
        self.nrelation = nrelation
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.hidden_dim = embed_dim
        self.eps = 2.0
        self.emb_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.eps) / self.hidden_dim]),
            requires_grad=False
        )
        self.relation_emb = nn.Embedding(nrelation, embed_dim)
        nn.init.uniform_(
            tensor=self.relation_emb.weight,
            a=-self.emb_range.item(),
            b=self.emb_range.item()
        )

        model_func = {
            'TransE': self.TransE,
            # 'DistMult': self.DistMult,
            # 'ComplEx': self.ComplEx,
            # 'RotatE': self.RotatE,
            # 'pRotatE': self.pRotatE
        }
        self.score_func = model_func[args.ke_model]

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=2, dim=2)
        return score

    def forward(self, heads, tails, nheads, ntails, heads_r, tails_r, relations,
                relation_desc_emb=None, conditioned_emb=None):
        """entity_r: Concatenation of the description for the entity h and the description for the relation r."""
        heads = heads[:, 0, :].unsqueeze(1)
        tails = tails[:, 0, :].unsqueeze(1)
        heads_r = heads_r[:, 0, :].unsqueeze(1)
        tails_r = tails_r[:, 0, :].unsqueeze(1)

        nheads = nheads[:, 0, :].view(heads.size(0), -1, self.hidden_dim)
        ntails = ntails[:, 0, :].view(tails.size(0), -1, self.hidden_dim)

        if relation_desc_emb is not None:  # Entity and Relation Descriptions as Embeddings, use relation descriptions
            relations = relation_desc_emb[:, 0, :].unsqueeze(1)
        else:
            print(relations)
            relations = self.relation_emb(relations)
            relations = relations.unsqueeze(1)

        heads = heads.type(torch.float32)
        tails = tails.type(torch.float32)
        nheads = nheads.type(torch.float32)
        ntails = ntails.type(torch.float32)
        heads_r = heads_r.type(torch.float32)
        tails_r = tails_r.type(torch.float32)
        relations = relations.type(torch.float32)

        if conditioned_emb is not None:  # Entity Embeddings Conditioned on Relations
            pos_scores = (self.score_func(heads_r, relations, tails) + self.score_func(heads, relations, tails_r)) / 2.0
        else:
            pos_scores = self.score_func(heads, relations, tails)
        neg_hscores = self.score_func(nheads, relations, tails_r)
        neg_tscores = self.score_func(heads_r, relations, ntails)
        neg_scores = torch.cat((neg_hscores, neg_tscores), dim=1)  # check the shape
        return pos_scores, neg_scores


class KEPLERClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.Tanh()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class KEPLERLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight=None):
        super().__init__()
        self.dense = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.decoder.bias = self.bias
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        # x = F.linear(x, self.weight) + self.bias
        x = self.decoder(x)
        return x


def KE_loss(pos_score, neg_score):
    """
    :return: ke_loss
    """
    pLoss = F.logsigmoid(pos_score).squeeze(dim=1)
    nLoss = F.logsigmoid(-neg_score).mean(dim=1)
    ke_loss = (-pLoss.mean() - nLoss.mean()) / 2.0
    return ke_loss


def MLM_loss(logits, targets, padding_idx=-100):
    """
    :param logits: logits = model(**sample['MLM']['net_input'])[0], sample['MLM']['net_input'] = tokenizer(...)
    :param targets: targets = sample['MLM']['targets']
    :param padding_idx: the index of '<pad>' in vocab
    :return: mlm_loss
    """
    logits = logits[:, -1, :]  # 取mask位置
    mlm_loss = F.nll_loss(
        F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        ),
        targets.view(-1),
        reduction='mean',
        ignore_index=padding_idx,
    )
    return mlm_loss


# For foward check
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default=2, type=int, required=True, help="sentence classsification task.")
    parser.add_argument("--base_model", default=None, type=str, required=True, help="base model.")
    parser.add_argument("--pooler_dropout", default=0.2, type=float, required=True, help="base model.")
    parser.add_argument("--gamma", default=0.2, type=float, required=True, help="gamma.")
    parser.add_argument("--nrelation", default=10, type=int, required=True, help="base model.")
    parser.add_argument("--ke_model", default='TransE', type=str, required=True, help="ke model.")
    parser.add_argument("--padding_idx", default=100, type=int, required=True, help="padding idx in vocab.")

    args = parser.parse_args()

    model = KEPLER(config=args)
    state_dict = torch.load('./roberta-base/pytorch_model.bin')
    model.lm_head.dense.weight = nn.Parameter(state_dict['lm_head.dense.weight'])
    model.lm_head.dense.bias = nn.Parameter(state_dict['lm_head.dense.bias'])
    model.lm_head.layer_norm.weight = nn.Parameter(state_dict['lm_head.layer_norm.weight'])
    model.lm_head.layer_norm.bias = nn.Parameter(state_dict['lm_head.layer_norm.bias'])
    model.lm_head.bias = nn.Parameter(state_dict['lm_head.bias'])
    model.lm_head.decoder.weight = nn.Parameter(state_dict['lm_head.decoder.weight'])
    model.lm_head.decoder.bias = nn.Parameter(state_dict['lm_head.decoder.bias'])
    # for name, parms in model.named_parameters():
    #     if 'lm_head.dense' in name:
    #         print('-->name:', name)
    #         print('-->param:', parms)

    batch = dict()
    sample_desc = "Fred Goodwins (26 February 1891 – April 1923) was an English actor, film director and" \
                  " screenwriter of the silent era. He appeared in 24 films between 1915 and 1921."
    tokenized_inputs = model.tokenizer(sample_desc, return_tensors="pt")
    batch['mlm_inputs'] = tokenized_inputs
    print(tokenized_inputs['input_ids'])
    batch['mlm_targets'] = torch.zeros(size=[1], dtype=torch.int64)
    batch['ke_targets'] = tokenized_inputs['input_ids']
    batch['classification_head'] = False

    batch['ke_inputs'] = dict()
    batch['ke_inputs']["heads"] = tokenized_inputs['input_ids']
    batch['ke_inputs']["tails"] = tokenized_inputs['input_ids']
    batch['ke_inputs']["nheads"] = tokenized_inputs['input_ids']
    batch['ke_inputs']["ntails"] = tokenized_inputs['input_ids']
    batch['ke_inputs']["heads_r"] = tokenized_inputs['input_ids']
    batch['ke_inputs']["tails_r"] = tokenized_inputs['input_ids']

    input_map = dict()
    input_map['mlm_inputs'] = batch['mlm_inputs']
    ke_inputs = batch['ke_inputs']
    mlm_targets = batch['mlm_targets']
    ke_targets = batch['ke_targets']

    loss = model.train_step(batch)

    print(loss.detach().cpu().numpy())
