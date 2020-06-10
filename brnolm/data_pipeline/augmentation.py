import torch


class Substitutor:
    def __init__(self, rate, replacements_range):
        self.rate = rate
        self.replacements_range = replacements_range
        if not isinstance(replacements_range, int) or replacements_range < 0:
            raise ValueError(f"Replacements range needs to be a positive integer, got {replacements_range}")

    def __call__(self, X, targets):
        replacements = torch.randint(0, self.replacements_range, X.shape, device=X.device, dtype=X.dtype)
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
    def __init__(self, source, substitution_rate=-1.0, replacements_range=None, deletion_rate=-1.0):
        self.source = source
        if substitution_rate > 0.0:
            self.substitutor = Substitutor(substitution_rate, replacements_range)
        else:
            self.substitutor = None

        if deletion_rate > 0.0:
            self.deletor = Deletor(deletion_rate)
        else:
            self.deletor = None

    def __iter__(self):
        for X, t in self.source:
            if self.deletor:
                X, t = self.deletor(X, t)
            if self.substitutor:
                X, t = self.substitutor(X, t)
            yield X, t
