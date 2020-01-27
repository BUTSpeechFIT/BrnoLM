import brnolm.data_pipeline.split_corpus_dataset as split_corpus_dataset

import torch
from .common import TestCase

from .utils import getStream


class TokenizedSplitTests(TestCase):
    def setUp(self):
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()

        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_word(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        tokens_string = next(iter(ts))
        expectation = (torch.LongTensor([0]), torch.LongTensor([1]))  # input, target
        self.assertEqual(tokens_string, expectation)

    def test_single_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        tokens_strings = list(iter(ts))
        expectation = [(torch.LongTensor([0]), torch.LongTensor([1])), (torch.LongTensor([1]), torch.LongTensor([2])), (torch.LongTensor([2]), torch.LongTensor([0]))]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_len(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        self.assertEqual(len(ts), len(self.test_words_short)-1)

    def test_len_no_output(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 5)
        self.assertEqual(len(ts), 0)

    def test_two_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [(torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))]
        self.assertEqual(tokens_strings, expectation)

    def test_two_word_seq_long(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [(torch.LongTensor([0, 1]), torch.LongTensor([1, 2])), (torch.LongTensor([2, 0]), torch.LongTensor([0, 0]))]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        words = list(ts.input_words())
        self.assertEqual(words, ['a', 'b', 'c'])  # we expect the input words

    def test_two_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 2)
        words = list(ts.input_words())
        self.assertEqual(words, ['a b'])  # we expect the input words


class TokenizedSplitSingleTargetTests(TestCase):
    def setUp(self):
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()

        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_word(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 1)
        tokens_string = next(iter(ts))
        expectation = (torch.LongTensor([0]), torch.LongTensor([1]))  # input, target
        self.assertEqual(tokens_string, expectation)

    def test_single_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 1)
        tokens_strings = list(iter(ts))
        expectation = [(torch.LongTensor([0]), torch.LongTensor([1])), (torch.LongTensor([1]), torch.LongTensor([2])), (torch.LongTensor([2]), torch.LongTensor([0]))]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_len(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 1)
        self.assertEqual(len(ts), len(self.test_words_short)-1)

    def test_len_no_output(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 5)
        self.assertEqual(len(ts), 0)

    def test_two_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1]), torch.LongTensor([2])),
            (torch.LongTensor([1, 2]), torch.LongTensor([0]))
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_two_word_seq_long(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1]), torch.LongTensor([2])),
            (torch.LongTensor([1, 2]), torch.LongTensor([0])),
            (torch.LongTensor([2, 0]), torch.LongTensor([0]))
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 1)
        words = list(ts.input_words())
        self.assertEqual(words, ['a', 'b', 'c'])  # we expect the input words

    def test_two_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplitSingleTarget(data_source, self.vocab, 2)
        words = list(ts.input_words())
        self.assertEqual(words, ['a b', 'b c'])  # we expect the input words


class DomainAdaptationSplitTests(TestCase):
    def setUp(self):
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()

        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_word(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 1, 0.5)
        tokens_string = next(iter(ts))
        expectation = (torch.LongTensor([0]), torch.LongTensor([1]))  # input, target
        self.assertEqual(tokens_string, expectation)

    def test_single_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 1, 0.5)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0]), torch.LongTensor([1])),
            (torch.LongTensor([1]), torch.LongTensor([2])),
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_len(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 1, 0.5)
        self.assertEqual(len(ts), 2)

    def test_len_no_output(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 3, 0.5)
        self.assertEqual(len(ts), 0)

    def test_two_word_seq(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 2, 0.5)
        tokens_strings = list(iter(ts))
        expectation = [(torch.LongTensor([0, 1]), torch.LongTensor([2]))]
        self.assertEqual(tokens_strings, expectation)

    def test_two_word_seq_long(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 2, 0.25)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1]), torch.LongTensor([2])),
            (torch.LongTensor([1, 2]), torch.LongTensor([0])),
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 1, end_portion=0.5)
        words = list(ts.input_words())
        self.assertEqual(words, ['a'])  # we expect the input words

    def test_two_word_retrieval(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplit(data_source, self.vocab, 2, 0.5)
        words = list(ts.input_words())
        self.assertEqual(words, ['a a'])  # we expect the input words


class DomainAdaptationSplitFFMultiTargetTests(TestCase):
    def setUp(self):
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()

        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_word(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 1, 1, end_portion=0.5)
        tokens_string = next(iter(ts))
        expectation = (torch.LongTensor([0]), torch.LongTensor([1]))  # input, target
        self.assertEqual(tokens_string, expectation)

    def test_single_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 1, 2, end_portion=0.5)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1]), torch.LongTensor([1, 2])),
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_len(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 1, 1, end_portion=0.5)
        self.assertEqual(len(ts), 2)

    def test_len_no_output(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 5, 1, end_portion=0.5)
        self.assertEqual(len(ts), 0)

    def test_two_word_seq_long_st(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 2, 1, end_portion=0.5)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1]), torch.LongTensor([2])),
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_two_word_seq_long_mt(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 2, 2, end_portion=0.25)
        tokens_strings = list(iter(ts))
        expectation = [
            (torch.LongTensor([0, 1, 2]), torch.LongTensor([2, 0])),
        ]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_retrieval(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 1, 1, end_portion=0.5)
        words = list(ts.input_words())
        self.assertEqual(words, ['a'])  # we expect the input words

    def test_two_word_retrieval(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.DomainAdaptationSplitFFMultiTarget(data_source, self.vocab, 2, 1, end_portion=0.5)
        words = list(ts.input_words())
        self.assertEqual(words, ['a a'])  # we expect the input words
