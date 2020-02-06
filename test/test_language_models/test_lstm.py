from test.common import TestCase

import torch

from brnolm.language_models.lstm_model import LSTMLanguageModel


class OutputExtractionTests(TestCase):
    def test_multilayer(self):
        model = LSTMLanguageModel(4, ninp=10, nhid=10, nlayers=2, dropout=0.0)
        h0 = model.init_hidden(3)
        o, h1 = model(torch.tensor([[1], [2], [3]]), h0)

        self.assertEqual(model.extract_output_from_h(h1).unsqueeze(1), o)
