from test.common import TestCase

import torch

from brnolm.language_models.lstm_model import LSTMLanguageModel
from brnolm.language_models.encoders import FlatEmbedding


class OutputExtractionTests(TestCase):
    def test_multilayer(self):
        encoder = FlatEmbedding(4, 10)
        model = LSTMLanguageModel(token_encoder=encoder, dim_input=10, dim_lstm=10, nb_layers=2, dropout=0.0)
        h0 = model.init_hidden(3)
        o, h1 = model(torch.tensor([[1], [2], [3]]), h0)

        self.assertEqual(model.extract_output_from_h(h1).unsqueeze(1), o)
