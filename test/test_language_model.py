import unittest

import torch

from brnolm.language_models.lstm_model import LSTMLanguageModel
from brnolm.language_models.decoders import FullSoftmaxDecoder
from brnolm.language_models.language_model import LanguageModel
from brnolm.language_models.vocab import Vocabulary


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 1)
        self.model_i = torch.nn.Embedding(4, 1)
        self.model_i.weight[0, 0] = -1
        self.model_i.weight[1, 0] = 1
        self.model_i.weight[2, 0] = 2
        self.model_i.weight[3, 0] = 3

    def forward(self, x, h):
        return self.model_i(x), h + x.shape[1]


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model_o = torch.nn.Linear(1, 4)
        self._model_o.weight[0, 0] = -100
        self._model_o.weight[1, 0] = 2
        self._model_o.weight[2, 0] = 0
        self._model_o.weight[3, 0] = 5
        self._model_o.bias[0] = -100
        self._model_o.bias[1] = 1
        self._model_o.bias[2] = 3
        self._model_o.bias[3] = -4

    def forward(self, hs):
        return self._model_o(hs)


class TorchFaceTests(unittest.TestCase):
    def setUp(self):
        vocab = Vocabulary('<unk>', 0)
        vocab.add_from_text('a b c')
        model = FakeModel()
        decoder = FakeDecoder()
        self.lm = LanguageModel(model, decoder, vocab)

    def test_single_sentence(self):
        x = torch.tensor([[1, 0]])
        h0 = torch.tensor([[1.0]])
        y, h2 = self.lm(x, h0)

        expected_h = torch.tensor([[3.0]])
        expected_y = torch.tensor([[
            [-100.0+(-100.0), 2.0+1.0, 0.0+3.0, 5.0+(-4.0)],
            [100.0+(-100.0), -2.0+1.0, 0.0+3.0, -5.0+(-4.0)],
        ]])
        self.assertTrue((y == expected_y).all())
        self.assertTrue((h2 == expected_h).all())

    def test_batch(self):
        x = torch.tensor([[1, 0], [2, 3]])
        h0 = torch.tensor([[1.0], [-2.0]])
        y, h2 = self.lm(x, h0)

        expected_h = torch.tensor([[3.0], [0.0]])
        expected_y = torch.tensor([
            [
                [-100.0+(-100.0), 2.0+1.0, 0.0+3.0, 5.0+(-4.0)],
                [100.0+(-100.0), -2.0+1.0, 0.0+3.0, -5.0+(-4.0)],
            ],
            [
                [-200.0+(-100.0), 4.0+1.0, 0.0+3.0, 10.0+(-4.0)],
                [-300.0+(-100.0), 6.0+1.0, 0.0+3.0, 15.0+(-4.0)],
            ],
        ])
        self.assertTrue((y == expected_y).all())
        self.assertTrue((h2 == expected_h).all())


class BatchNLLCorrectnessTests(unittest.TestCase):
    def setUp(self):
        vocab = Vocabulary('<unk>', 0)
        vocab.add_from_text('a b c')
        model = LSTMLanguageModel(len(vocab), ninp=10, nhid=10, nlayers=2, dropout=0.0)
        decoder = FullSoftmaxDecoder(10, len(vocab))
        self.lm = LanguageModel(model, decoder, vocab)

    def test_no_input(self):
        self.assertEqual(self.lm.batch_nll([], (['a useless prefix'])), [])

    def test_single_sentence(self):
        sentence = 'aaa'
        prefix = None
        single_sentence_nll = self.lm.single_sentence_nll(list(sentence), prefix)
        self.assertEqual(self.lm.batch_nll([list(sentence)], prefix), [single_sentence_nll])

    def test_several_sentences(self):
        sentences = ['ab', 'aaab', 'aaca', 'cacc']
        batch = [list(s) for s in sentences]
        prefix = None

        target = [self.lm.single_sentence_nll(s, prefix) for s in sentences]
        self.assertEqual(self.lm.batch_nll(batch, prefix), target)

    def test_several_sentences_with_prefix(self):
        sentences = ['ab', 'aaab', 'aaca', 'cacc']
        batch = [list(s) for s in sentences]
        prefix = 'c'

        target = [self.lm.single_sentence_nll(s, prefix) for s in sentences]
        self.assertEqual(self.lm.batch_nll(batch, prefix), target)
