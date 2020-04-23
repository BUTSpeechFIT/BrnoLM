import torch


class Corruptor:
    def __init__(self, source, rate, replacements_range):
        self.source = source
        self.rate = rate
        self.replacement_range = replacements_range

    def __iter__(self):
        for X, targets in self.source:
            replacements = torch.randint(0, self.replacement_range, X.shape, device=X.device, dtype=X.dtype)
            mask = torch.full(X.shape, self.rate, device=X.device)
            mask = torch.bernoulli(mask).long()

            yield X * (1-mask) + replacements * mask, targets
