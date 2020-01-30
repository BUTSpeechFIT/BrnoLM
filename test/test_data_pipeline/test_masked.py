from test.common import TestCase
from torch import tensor


from brnolm.data_pipeline.masked import masked_tensor_from_sentences


class MaskedDataCreationTests(TestCase):
    def test_requires_sequence(self):
        self.assertRaises(ValueError, masked_tensor_from_sentences, 0)

    def test_requires_batch(self):
        self.assertRaises(ValueError, masked_tensor_from_sentences, [0])

    def test_single_sentence_single_step(self):
        x, t, m = masked_tensor_from_sentences([[0, 1]])

        e_input = tensor([
            [0],
        ])
        e_target = tensor([
            [1],
        ])
        e_mask = tensor([
            [1],
        ])

        self.assertEqual(x, e_input)
        self.assertEqual(t, e_target)
        self.assertEqual(m, e_mask)

    def test_single_sentence_multiple_steps(self):
        x, t, m = masked_tensor_from_sentences([[0, 1, 2, 3]])

        e_input = tensor([
            [0, 1, 2],
        ])
        e_target = tensor([
            [1, 2, 3],
        ])
        e_mask = tensor([
            [1, 1, 1],
        ])

        self.assertEqual(x, e_input)
        self.assertEqual(t, e_target)
        self.assertEqual(m, e_mask)

    def test_multiple_sentences_matching(self):
        sentences = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
        x, t, m = masked_tensor_from_sentences(sentences)

        e_input = tensor([
            [0, 1, 2],
            [4, 5, 6],
        ])
        e_target = tensor([
            [1, 2, 3],
            [5, 6, 7],
        ])
        e_mask = tensor([
            [1, 1, 1],
            [1, 1, 1],
        ])

        self.assertEqual(x, e_input)
        self.assertEqual(t, e_target)
        self.assertEqual(m, e_mask)

    def test_multiple_sentences_nonmatching(self):
        sentences = [
            [0, 1, 2, 3],
            [4, 5, 6],
        ]
        x, t, m = masked_tensor_from_sentences(sentences)

        e_input = tensor([
            [0, 1, 2],
            [4, 5, 0],
        ])
        e_target = tensor([
            [1, 2, 3],
            [5, 6, 0],
        ])
        e_mask = tensor([
            [1, 1, 1],
            [1, 1, 0],
        ])

        self.assertEqual(x, e_input)
        self.assertEqual(t, e_target)
        self.assertEqual(m, e_mask)

    def test_multiple_sentences_nonmatching_first_short(self):
        sentences = [
            [4, 5, 6],
            [0, 1, 2, 3],
        ]
        x, t, m = masked_tensor_from_sentences(sentences)

        e_input = tensor([
            [4, 5, 0],
            [0, 1, 2],
        ])
        e_target = tensor([
            [5, 6, 0],
            [1, 2, 3],
        ])
        e_mask = tensor([
            [1, 1, 0],
            [1, 1, 1],
        ])

        self.assertEqual(x, e_input)
        self.assertEqual(t, e_target)
        self.assertEqual(m, e_mask)
