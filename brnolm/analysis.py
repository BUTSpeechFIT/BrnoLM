import torch


def categorical_entropy(p, eps=1e-100):
    zeros = p <= eps

    log_p = p.log()
    log_p.masked_fill_(zeros, 0.0) # eliminates -inf for p[x] = 0.0

    H_p = - torch.sum(p*log_p, dim=-1)
    return H_p / torch.log(torch.FloatTensor([2]))
    

def categorical_cross_entropy(p, q, eps=1e-100):
    zeros = p <= eps

    log_q = q.log()
    log_q = torch.zeros_like(p) + log_q
    log_q.masked_fill_(zeros, 0.0) # eliminates -inf for p[x] = 0.0

    Xent = - torch.sum(p*log_q, dim=-1)
    return Xent / torch.log(torch.FloatTensor([2]))


def categorical_kld(p, q):
    return categorical_cross_entropy(p, q) - categorical_entropy(p)
