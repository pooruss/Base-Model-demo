import torch
import torch.nn.functional as F


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
