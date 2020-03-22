from test.common import TestCase
import math
import torch

from brnolm.runtime.evaluation import get_oov_additional_cost
from brnolm.runtime.evaluation import OovCostApplicator


class OovCostTests(TestCase):
    def test_simple(self):
        oov_cost = get_oov_additional_cost(100, 1000)
        expected = -math.log(1.0/900)
        self.assertEqual(oov_cost, expected)


class OovCostApplicatorTests(TestCase):
    def setUp(self):
        self.cost_applier = OovCostApplicator(1.0, 0)

    def test_void(self):
        line_ids = torch.tensor([2, 1])
        losses = torch.tensor([0.5, 0.2])
        self.assertEqual(self.cost_applier(line_ids, losses), losses)

    def test_single_line_single_oov(self):
        line_ids = torch.tensor([2, 0, 1])
        losses = torch.tensor([0.5, 0.7, 0.2])
        adjusted_losses = torch.tensor([0.5, 1.7, 0.2])
        self.assertEqual(self.cost_applier(line_ids, losses), adjusted_losses)

    def test_single_line_multiple_oovs(self):
        line_ids = torch.tensor([0, 2, 0, 1])
        losses = torch.tensor([0.3, 0.5, 0.7, 0.2])
        adjusted_losses = torch.tensor([1.3, 0.5, 1.7, 0.2])
        self.assertEqual(self.cost_applier(line_ids, losses), adjusted_losses)

    def test_non_matching_len(self):
        line_ids = torch.tensor([0, 2, 1])
        losses = torch.tensor([0.3, 0.5, 0.7, 0.0])
        adjusted_losses = torch.tensor([1.3, 0.5, 0.7, 0.0])
        self.assertEqual(self.cost_applier(line_ids, losses), adjusted_losses)

    def test_zero_penalty(self):
        zero_applier = OovCostApplicator(0.0, 0)
        line_ids = torch.tensor([0, 2, 1])
        losses = torch.tensor([0.3, 0.5, 0.7, 0.0])
        adjusted_losses = torch.tensor([0.3, 0.5, 0.7, 0.0])
        self.assertEqual(zero_applier(line_ids, losses), adjusted_losses)
