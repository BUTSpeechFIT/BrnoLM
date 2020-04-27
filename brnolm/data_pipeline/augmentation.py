import torch


class Substitutor:
    def __init__(self, rate, replacements_range):
        self.rate = rate
        self.replacement_range = replacements_range

    def __call__(self, X, targets):
        replacements = torch.randint(0, self.replacement_range, X.shape, device=X.device, dtype=X.dtype)
        mask = torch.full(X.shape, self.rate, device=X.device)
        mask = torch.bernoulli(mask).long()

        return X * (1-mask) + replacements * mask, targets


class Corruptor:
    def __init__(self, source, rate, replacements_range):
        self.source = source
        self.substitutor = Substitutor(rate, replacements_range)

    def __iter__(self):
        for X, targets in self.source:
            yield self.substitutor(X, targets)
