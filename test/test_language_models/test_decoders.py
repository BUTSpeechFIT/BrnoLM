from test.common import TestCase
import torch

from brnolm.language_models.decoders import FullSoftmaxDecoder


class FullSoftmaxDecoderTests(TestCase):
    def test_raw_log_prob_shape(self):
        decoder = FullSoftmaxDecoder(4, 3)
        o = torch.zeros((2, 3, 4), dtype=torch.float64)
        t = torch.tensor([
            [0, 1, 2],
            [2, 1, 1],
        ])
        y = decoder.neg_log_prob_raw(o, t)

        self.assertEqual(y.shape, (2, 3))
