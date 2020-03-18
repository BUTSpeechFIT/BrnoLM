from unittest import TestCase
import math

from brnolm.runtime.evaluation import get_oov_additional_cost


class OovCostTests(TestCase):
    def test_simple(self):
        oov_cost = get_oov_additional_cost(100, 1000)
        expected = -math.log(1.0/900)
        self.assertEqual(oov_cost, expected)
