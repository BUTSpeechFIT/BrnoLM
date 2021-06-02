import torch
import torch.nn as nn
from typing import Tuple


class LSTMLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, token_encoder, dim_input, dim_lstm, nb_layers, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = token_encoder
        self.rnn = nn.LSTM(dim_input, dim_lstm, nb_layers, dropout=dropout, batch_first=True)

        self.dim_lstm = dim_lstm
        self.nlayers = nb_layers
        self.nhid = dim_lstm

        self.batch_first = True
        self.in_len = 1

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden: Tuple[torch.Tensor, torch.Tensor]):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output, hidden

    def output_expected_embs(self, input):
        assert (len(input.size()) == 2)  # batch X time index
        assert (input.size()[0] == 1)

        hidden = self.init_hidden(1)
        emb = self.drop(self.encoder(input))
        outputs, _ = self.rnn(emb, hidden)
        return outputs

    @torch.jit.export
    def init_hidden(self, bsz: int):
        weight = self.state_dict()['rnn.weight_ih_l0']
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    @torch.jit.export
    def extract_output_from_h(self, hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        h = hidden_state[0]  # hidden state is (h, c)
        return h[-1]  # last layer is the output one


class LSTMPLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(nhid, ninp)

        if tie_weights:
            raise NotImplementedError

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

        self.batch_first = True
        self.in_len = 1

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        projected = self.proj(output)
        return projected, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
