import torch.nn as nn


class FlatEmbedding(nn.Module):
    def __init__(self, nb_tokens, dim_embs, init_range=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(nb_tokens, dim_embs)
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)

    def forward(self, x):
        return self.embeddings(x)
