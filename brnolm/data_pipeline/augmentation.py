import torch


class Corruptor:
    def __init__(self, source, rate):
        self.source = source
        self.rate = rate

    def __iter__(self):
        for X, targets in self.source:
            replacements = torch.randint(0, 1000, X.shape, device=X.device, dtype=X.dtype)
            mask = torch.full(X.shape, self.rate, device=X.device)
            mask = torch.bernoulli(mask).long()

            yield X * (1-mask) + replacements * mask, targets
