from .common import TestCase
import os
import sys
import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

import brnolm.data_pipeline.split_corpus_dataset as split_corpus_dataset
import brnolm.smm_itf.ivec_appenders as ivec_appenders
from brnolm.language_models.vocab import Vocabulary

from .utils import getStream

try:
    import brnolm.smm_itf.smm_ivec_extractor as smm_ivec_extractor
    from smm import SMM, estimate_ubm
except ImportError:
    sys.stderr.write('Failed to import SMM implementation\n')


@unittest.skipIf(os.environ.get('TEST_SMM') != 'yes', "For SMM tests, set TEST_SMM='yes'")
class CheatingIvecAppenderTests(TestCase):
    def setUp(self):
        self.ivec_eetor = lambda x: np.asarray([hash(x) % 1337])
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()
        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_data(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        seqs = next(iter(appender))
        first = seqs[2]

        self.assertEqual(first, expectation)

    def test_whole_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        ]

        seqs = [x[2] for x in (iter(appender))]
        self.assertEqual(seqs, expectation)

    def test_whole_seq_with_next(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]
        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        ]

        seq0 = next(appender)[2]
        self.assertEqual(seq0, expectation[0])

        seq1 = next(appender)[2]
        self.assertEqual(seq1, expectation[1])

        seq2 = next(appender)[2]
        self.assertEqual(seq2, expectation[2])

    def test_iter_ends(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

        next(appender)
        next(appender)
        next(appender)

        self.assertRaises(StopIteration, next, appender)


@unittest.skipIf(os.environ.get('TEST_SMM') != 'yes', "For SMM tests, set TEST_SMM='yes'")
class HistoryIvecAppenderTests(TestCase):
    def setUp(self):
        self.ivec_eetor = lambda x: np.asarray([hash(x) % 1337])
        self.test_words_short = "a b c a".split()
        self.test_words_long = "ab bb cd a b".split()
        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2,
            "ab": 3,
            "bb": 4,
            "cd": 5,
        }

    def test_single_data(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = self.ivec_eetor(" ".join([]))
        seqs = next(iter(appender))
        first = seqs[2]

        self.assertEqual(first, expectation)

    def test_whole_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:0])),
            self.ivec_eetor(" ".join(self.test_words_short[:1])),
            self.ivec_eetor(" ".join(self.test_words_short[:2])),
        ]
        seqs = [x[2] for x in (iter(appender))]

        self.assertEqual(seqs, expectation)

    def test_multiletter_words(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

        # cannot acces ts._tokens, it's an implementation
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_long[:0])),
            self.ivec_eetor(" ".join(self.test_words_long[:1])),
            self.ivec_eetor(" ".join(self.test_words_long[:2])),
            self.ivec_eetor(" ".join(self.test_words_long[:3])),
        ]
        seqs = [x[2] for x in iter(appender)]

        self.assertEqual(seqs, expectation)


@unittest.skipIf(os.environ.get('TEST_SMM') != 'yes', "For SMM tests, set TEST_SMM='yes'")
class ParalelIvecAppenderTests(TestCase):
    def setUp(self):
        documents = ["a set of documents containing eigth different", "words of documents"]
        cvect = CountVectorizer(documents, strip_accents='ascii', analyzer='word')
        cvect.fit(documents)
        vocab = cvect.get_feature_names()
        self.cvect = CountVectorizer(documents, strip_accents='ascii', analyzer='word', vocabulary=vocab)

        bows = cvect.transform(documents)
        bows = bows.A.astype(np.float32).T

        ubm = estimate_ubm(bows)
        hyper = {
            'lam_w': 0.1,
            'reg_t': 0.1,
            'lam_t': 0.1,
            'iv_dim': 4
        }
        smm = SMM(bows.shape[0], ubm, hyper)

        self.extractor = smm_ivec_extractor.IvecExtractor(smm, nb_iters=10, lr=0.1, tokenizer=self.cvect)

        self.vocab = Vocabulary(unk_word="<unk>", unk_index=0)
        self.vocab.add_from_text(" ".join(documents))

        self.translator = self.extractor.build_translator(self.vocab)

    def test_single_data(self):
        ws = [self.vocab[w] for w in self.vocab]

        xs = [
            torch.LongTensor([[ws[0], ws[1]], [ws[0], ws[2]]]),
            torch.LongTensor([[ws[2], ws[3]], [ws[4], ws[6]]]),
            torch.LongTensor([[ws[4], ws[5]], [ws[0], ws[1]]]),
            torch.LongTensor([[ws[2], ws[3]], [ws[0], ws[4]]]),
        ]

        ts = [
            torch.LongTensor([[ws[1], ws[2]], [ws[1], ws[3]]]),
            torch.LongTensor([[ws[3], ws[4]], [ws[5], ws[7]]]),
            torch.LongTensor([[ws[5], ws[6]], [ws[1], ws[2]]]),
            torch.LongTensor([[ws[3], ws[4]], [ws[1], ws[5]]]),
        ]

        masks = [
            torch.LongTensor([]),
            torch.LongTensor([0, 1]),
            torch.LongTensor([0]),
            torch.LongTensor([1]),
        ]

        stream = zip(xs, ts, masks)
        ivec_appender = ivec_appenders.ParalelIvecAppender(stream, self.extractor, self.translator)

        exp_ivecs = [
            self.extractor(self.extractor.zero_bows(2)),
            self.extractor(self.translator(xs[0])),
            self.extractor(torch.stack([
                self.translator(torch.cat([xs[0][0], xs[1][0]])),
                self.extractor.zero_bows(1).squeeze()
            ])),
            self.extractor(torch.stack([
                self.translator(xs[2][1]),
                self.extractor.zero_bows(1).squeeze(),
            ])),
        ]

        exp_stream = list(zip(xs, ts, exp_ivecs, masks))
        obs_stream = list(ivec_appender)

        self.assertEqual(obs_stream[0], exp_stream[0])
        self.assertEqual(obs_stream[1], exp_stream[1])
        self.assertEqual(obs_stream[2], exp_stream[2])
        self.assertEqual(obs_stream[3], exp_stream[3])
