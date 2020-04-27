import torch


class Substitutor:
    def __init__(self, rate, replacements_range):
        self.rate = rate
        self.replacement_range = replacements_range
        if not isinstance(replacements_range, int) or replacements_range < 0:
            raise ValueError(f"Replacements range needs to be a positive integer, got {replacements_range}")

    def __call__(self, X, targets):
        replacements = torch.randint(0, self.replacement_range, X.shape, device=X.device, dtype=X.dtype)
        mask = torch.full(X.shape, self.rate, device=X.device)
        mask = torch.bernoulli(mask).long()

        return X * (1-mask) + replacements * mask, targets


class Deletor:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, X, targets):
        timemask = torch.full((X.shape[1], ), 1 - self.rate, device=X.device)
        timemask = torch.bernoulli(timemask).bool()
        return X[:, timemask], targets[:, timemask]


class Corruptor:
    def __init__(self, source, rate, replacements_range):
        self.source = source
        self.substitutor = Substitutor(rate, replacements_range)
        self.deletor = Deletor(rate)

    def __iter__(self):
        for X, targets in self.source:
            X_out, t_out = self.deletor(X, targets)
            X_out, t_out = self.substitutor(X_out, t_out)
            yield X_out, t_out
