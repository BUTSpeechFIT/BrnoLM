import unittest
from .common import TestCase
import os
import sys

try:
    import brnolm.smm_itf.smm_ivec_extractor as smm_ivec_extractor
except ImportError:
    sys.stderr.write('Failed to import SMM implementation\n')
from brnolm.language_models.vocab import Vocabulary

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


class DummySMM(nn.Module):
    def __init__(self, ivec_dim):
        self.W = Variable(torch.zeros(ivec_dim, 1), requires_grad=True)
        self.T = Variable(torch.zeros(ivec_dim, 20), requires_grad=True)


@unittest.skipIf(os.environ.get('TEST_SMM') != 'yes', "For SMM tests, set TEST_SMM='yes'")
class IvecExtractorTests(TestCase):
    def setUp(self):
        self.documents_six = ["text consisting of SIX different words", "text"]

        # note that following also contains puctuation marks, to be ignored by CountVectorizer
        self.documents_seven = ["text consisting of SEVEN different words", ", seventh ."]

    def build_neededs(self, documents):
        smm = DummySMM(ivec_dim=4)
        cvect = CountVectorizer(documents, strip_accents='ascii', analyzer='word')
        cvect.fit(documents)
        vocab = cvect.get_feature_names()
        self.cvect = CountVectorizer(documents, strip_accents='ascii', analyzer='word', vocabulary=vocab)

        self.extractor = smm_ivec_extractor.IvecExtractor(smm, nb_iters=10, lr=0.1, tokenizer=self.cvect)
        self.vocab = Vocabulary(unk_word="<unk>", unk_index=0)
        self.vocab.add_from_text(" ".join(documents))

    def test_one_empty_bow(self):
        self.build_neededs(self.documents_six)

        ivecs = self.extractor.zero_bows(1)
        expectation = torch.zeros(1, 6)
        self.assertEqual(ivecs, expectation)

    def test_two_empty_bows(self):
        self.build_neededs(self.documents_six)
        ivecs = self.extractor.zero_bows(2)
        expectation = torch.zeros(2, 6)
        self.assertEqual(ivecs, expectation)

    def test_build_translator_single_word_translation(self):
        self.build_neededs(self.documents_six)
        translator = self.extractor.build_translator(self.vocab)
        word = "SIX"
        lm_word = torch.LongTensor([self.vocab[word]])
        cv_word = torch.from_numpy(self.cvect.transform([word]).A.astype(np.float32)).squeeze()
        self.assertEqual(translator(lm_word), cv_word)

    def test_build_translator_two_word_translation(self):
        self.build_neededs(self.documents_six)
        translator = self.extractor.build_translator(self.vocab)
        words = "SIX words"
        lm_words = torch.LongTensor([self.vocab[w] for w in words.split()])
        cv_words = torch.from_numpy(self.cvect.transform([words]).A.astype(np.float32)).squeeze()
        self.assertEqual(translator(lm_words), cv_words)

    def test_build_translator_two_two_word_translations(self):
        self.build_neededs(self.documents_six)
        translator = self.extractor.build_translator(self.vocab)
        words = ["SIX words", "of text"]
        lm_words = torch.LongTensor([[self.vocab[w] for w in seq.split()] for seq in words])
        cv_words = torch.from_numpy(self.cvect.transform(words).A.astype(np.float32)).squeeze()
        self.assertEqual(translator(lm_words), cv_words)

    def test_build_translator_two_two_word_translations_different_vocab(self):
        self.build_neededs(self.documents_seven)
        translator = self.extractor.build_translator(self.vocab)
        words = ["SIX words", "of text"]
        lm_words = torch.LongTensor([[self.vocab[w] for w in seq.split()] for seq in words])
        cv_words = torch.from_numpy(self.cvect.transform(words).A.astype(np.float32)).squeeze()
        self.assertEqual(translator(lm_words), cv_words)
