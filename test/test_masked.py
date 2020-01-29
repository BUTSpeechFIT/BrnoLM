from .common import TestCase
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
