import torch


class FullSoftmaxDecoder(torch.nn.Module):
    def __init__(self, nb_hidden, nb_output, init_range=0.1):
        super().__init__()

        self.projection = torch.nn.Linear(nb_hidden, nb_output)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.projection.weight.data.uniform_(-init_range, init_range)
        self.projection.bias.data.fill_(0)

        self.nllloss = torch.nn.NLLLoss(reduction='sum')

    def forward(self, X):
        a = self.projection(X)
        return self.log_softmax(a)

    def neg_log_prob_raw(self, X, targets):
        orig_shape = targets.shape
        preds = self.forward(X)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1, preds.size(-1))

        return torch.nn.functional.nll_loss(preds_flat, targets_flat, reduction='none').view(orig_shape)

    @torch.jit.export
    def neg_log_prob(self, X, targets):
        return self.neg_log_prob_raw(X, targets).sum(), targets.numel()


class CustomLossFullSoftmaxDecoder(torch.nn.Module):
    def __init__(self, nb_hidden, nb_output, init_range=0.1, label_smoothing=None):
        super().__init__()

        assert label_smoothing is None or (label_smoothing >= 0.0 and label_smoothing < 1.0)

        self.projection = torch.nn.Linear(nb_hidden, nb_output)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.projection.weight.data.uniform_(-init_range, init_range)
        self.projection.bias.data.fill_(0)

        if label_smoothing is not None:
            self.core_loss = LabelSmoothedNLLLoss(label_smoothing)
        else:
            self.core_loss = plain_nll_loss

    def forward(self, X):
        a = self.projection(X)
        return self.log_softmax(a)

    def neg_log_prob_raw(self, X, targets):
        orig_shape = targets.shape
        preds = self.forward(X)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1, preds.size(-1))

        return self.core_loss(preds_flat, targets_flat).view(orig_shape)

    def neg_log_prob(self, X, targets):
        return self.neg_log_prob_raw(X, targets).sum(), targets.numel()


def plain_nll_loss(preds, targets):
    return torch.nn.functional.nll_loss(preds, targets, reduction='none')


class LabelSmoothedNLLLoss:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, preds, targets):
        eps = self.amount
        n_class = preds.size(1)

        assert len(targets.shape) == 1
        assert len(preds.shape) == 2

        smooth_targets = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = smooth_targets * (1 - eps) + (1 - smooth_targets) * eps / n_class
        loss = -(smooth_targets * preds).sum(axis=1)

        return loss
