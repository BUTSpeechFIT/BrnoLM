import torch
import torch.nn as nn
import torch.nn.functional as F


class BengioModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, emb_size, in_len, nb_hidden, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emb_size)
        self.emb2h = nn.ModuleList([nn.Linear(emb_size, nb_hidden) for _ in range(in_len)])

        self.init_weights()

        self.nb_hidden = nb_hidden
        self.in_len = in_len
        self.emb_size = emb_size

        self.batch_first = True

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for e2h in self.emb2h:
            e2h.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        projections = [proj(emb[:, i:emb.size(1)-(self.in_len-i)+1]) for i, proj in enumerate(self.emb2h)]
        projections = torch.stack(projections, dim=-1)
        output = F.tanh(torch.sum(projections, dim=-1))
        output = self.drop(output)
        return output, hidden

    def init_hidden(self, bsz):
        # not used, but to fit into the framework of other ivec-LMs
        weight = next(self.parameters()).data
        return weight.new_zeros(1, bsz, self.nb_hidden)


class BengioModelIvecInput(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, emb_size, in_len, nb_hidden, dropout, ivec_dim):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emb_size)
        self.emb2h = nn.ModuleList([nn.Linear(emb_size, nb_hidden) for _ in range(in_len)])
        self.ivec2h = nn.Linear(ivec_dim, nb_hidden)

        self.init_weights()

        self.nb_hidden = nb_hidden
        self.in_len = in_len
        self.emb_size = emb_size

        self.batch_first = True

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for e2h in self.emb2h:
            e2h.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        if len(ivec.size()) == 1:
            ivec = ivec.unsqueeze(0)
        emb = self.drop(self.encoder(input))
        projections = [proj(emb[:, i:emb.size(1)-(self.in_len-i)+1]) for i, proj in enumerate(self.emb2h)]
        nb_timesteps = projections[0].size(1)
        projected_ivec = self.ivec2h(ivec).unsqueeze(dim=-2).expand(-1, nb_timesteps, -1)
        projections = torch.stack(projections + [projected_ivec], dim=-1)
        output = F.tanh(torch.sum(projections, dim=-1))
        output = self.drop(output)
        return output, hidden

    def init_hidden(self, bsz):
        # not used, but to fit into the framework of other ivec-LMs
        weight = next(self.parameters()).data
        return weight.new_zeros(1, bsz, self.nb_hidden)
