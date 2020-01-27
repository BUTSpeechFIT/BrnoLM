import unittest
import brnolm.language_models.vocab as vocab
from io import StringIO

from collections import Mapping


class IndexGeneratorTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_zero_start(self):
        ig = vocab.IndexGenerator([])
        self.assertEqual(ig.next(), 0)

    def test_respects_assigned(self):
        ig = vocab.IndexGenerator([0, 1])
        self.assertEqual(ig.next(), 2)

    def test_respects_sparse_assigned(self):
        ig = vocab.IndexGenerator([0, 2, 3])
        nexts = [ig.next() for i in range(2)]
        self.assertEqual(nexts, [1, 4])


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.example_line = "hello world of vocabularies !"

    def test_getitem(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        self.assertEqual(vocabulary['<unk>'], 0)

    def test_is_mapping(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        self.assertTrue(isinstance(vocabulary, Mapping))

    def test_getitem_for_unknown(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        self.assertEqual(vocabulary['a'], 0)

    def test_addition_un_unks(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        self.assertEqual(vocabulary['world'], 0)

        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], 0)

    def test_addition_is_unique(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['!'])
        self.assertNotEqual(vocabulary['world'], vocabulary['!'])

    def test_lenght_of_empty(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        self.assertEqual(len(vocabulary), 1)

    def test_lenght_after_addition(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        vocabulary.add_from_text(self.example_line)
        self.assertEqual(len(vocabulary), 6)

    def test_add_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        vocabulary.add_word('hi')
        self.assertEqual(len(vocabulary), 2)

    def test_unk_word(self):
        vocabulary = vocab.Vocabulary('<bla>', 1)
        vocabulary.add_word('hi')
        self.assertEqual(vocabulary.unk_word, '<bla>')

    def test_unk_ind(self):
        vocabulary = vocab.Vocabulary('<bla>', 2)
        vocabulary.add_word('hi')
        self.assertEqual(vocabulary.unk_ind, 2)

    def test_add_already_known_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        vocabulary.add_word('<unk>')
        self.assertEqual(len(vocabulary), 1)

    def test_backward_translation_consistent(self):
        vocabulary = vocab.Vocabulary('<unk>', 0)
        vocabulary.add_word('hi')
        self.assertEqual(vocabulary.i2w(vocabulary['hi']), 'hi')

    def test_continuity_test_positive(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 1
                                    b 2 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertTrue(vocabulary.is_continuous())

    def test_continuity_test_negative(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 1
                                    b 3 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertFalse(vocabulary.is_continuous())

    def test_missing_indexes_none(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 1
                                    b 2 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(vocabulary.missing_indexes(), [])

    def test_missing_indexes_middle(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 1
                                    b 3 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(vocabulary.missing_indexes(), [2])

    def test_missing_indexes_beginning(self):
        kaldi_vocab = StringIO(""" <unk> 1
                                    a 2
                                    b 3 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(vocabulary.missing_indexes(), [0])


class VocabFromKaldiTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_simple(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 1
                                    b 2 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(len(vocabulary), 3)
        self.assertEqual(vocabulary['<unk>'], 0)
        self.assertEqual(vocabulary['a'], 1)
        self.assertEqual(vocabulary['b'], 2)
        self.assertEqual(vocabulary['nonexistent'], 0)

    def test_nonzero_unk(self):
        kaldi_vocab = StringIO(""" a 0
                                    <unk> 1
                                    b 2 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(len(vocabulary), 3)
        self.assertEqual(vocabulary['<unk>'], 1)
        self.assertEqual(vocabulary['a'], 0)
        self.assertEqual(vocabulary['b'], 2)
        self.assertEqual(vocabulary['nonexistent'], 1)

    def test_unk_not_present(self):
        kaldi_vocab = StringIO(""" a 0
                                    b 1 """)
        with self.assertRaises(ValueError):
            vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")

    def test_non_continuous(self):
        kaldi_vocab = StringIO(""" <unk> 0
                                    a 3
                                    b 7 """)
        vocabulary = vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
        self.assertEqual(len(vocabulary), 3)
        self.assertEqual(vocabulary['<unk>'], 0)
        self.assertEqual(vocabulary['a'], 3)
        self.assertEqual(vocabulary['b'], 7)
        self.assertEqual(vocabulary['nonexistent'], 0)

    def test_malformed_line(self):
        kaldi_vocab = StringIO(""" a 0 junk
                                    <unk> 1
                                    b 2 """)
        with self.assertRaises(ValueError):
            vocab.vocab_from_kaldi_wordlist(kaldi_vocab, "<unk>")
