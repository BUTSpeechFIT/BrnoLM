import torch.nn as nn


class FlatEmbedding(nn.Module):
    def __init__(self, nb_tokens, dim_embs):
        super().__init__()
        self.embeddings = nn.Embedding(nb_tokens, dim_embs)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        return self.embeddings(x)
