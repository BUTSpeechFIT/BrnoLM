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
        preds = self.forward(X)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1, preds.size(-1))

        return torch.nn.functional.nll_loss(preds_flat, targets_flat, reduction='none')

    def neg_log_prob(self, X, targets):
        return self.neg_log_prob_raw(X, targets).sum(), targets.numel()
