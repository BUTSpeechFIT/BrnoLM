import torch
from .common import TestCase

from brnolm.runtime.runtime_utils import repackage_hidden


class TensorReorganizerTests(TestCase):
    def setUp(self):
        tensor = torch.tensor(
            [[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]],
            requires_grad=True
        )
        self.computed_tensor = tensor * 3

    def test_data_kept(self):
        repackaged = repackage_hidden(self.computed_tensor)
        self.assertEqual(self.computed_tensor, repackaged)

    def test_result_requires_grad(self):
        repackaged = repackage_hidden(self.computed_tensor)
        self.assertTrue(repackaged.requires_grad_)

    def test_is_detached(self):
        self.assertFalse(self.computed_tensor.grad_fn is None)
        repackaged = repackage_hidden(self.computed_tensor)
        self.assertTrue(repackaged.grad_fn is None)
