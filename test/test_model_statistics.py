from unittest import TestCase

from brnolm.runtime.model_statistics import scaled_int_str


class ScaledIntRepreTests(TestCase):
    def test_order_0(self):
        self.assertEqual(scaled_int_str(0), '0')

    def test_order_1(self):
        self.assertEqual(scaled_int_str(10), '10')

    def test_order_2(self):
        self.assertEqual(scaled_int_str(210), '210')

    def test_order_3(self):
        self.assertEqual(scaled_int_str(3210), '3.2k')

    def test_order_4(self):
        self.assertEqual(scaled_int_str(43210), '43.2k')

    def test_order_5(self):
        self.assertEqual(scaled_int_str(543210), '543.2k')

    def test_order_6(self):
        self.assertEqual(scaled_int_str(6543210), '6.5M')
