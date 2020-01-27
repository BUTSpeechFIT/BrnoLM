import unittest
from brnolm.language_models.lstm_model import LSTMLanguageModel
from brnolm.language_models.decoders import FullSoftmaxDecoder
from brnolm.language_models.language_model import LanguageModel
from brnolm.language_models.vocab import Vocabulary


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
