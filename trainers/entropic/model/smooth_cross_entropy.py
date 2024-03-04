import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


'''
def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    # Check if gold is a tuple (indicating CutMix targets)
    if isinstance(gold, tuple):
        gold1, gold2, lam = gold
        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold1.unsqueeze(1), value=(1.0 - smoothing) * lam)
        one_hot.scatter_add_(dim=1, index=gold2.unsqueeze(1), src=(1.0 - smoothing) * (1 - lam))
    else:
        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)

    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
'''
