from unittest import TestCase

from brnolm.oov_clustering.det import det_points_from_score_tg
from brnolm.oov_clustering.det import subsample_list
from brnolm.oov_clustering.det import area_under_curve, eer


class DetPointTests(TestCase):
    def test_trivial(self):
        score_tg = [
            (0.0, 0),
            (1.0, 1)
        ]

        det_points = [
            [0.5, 0.0],
            [0.0, 0.0],
            [0.0, 0.5],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg)[0], det_points)

    def test_permutation(self):
        score_tg = [
            (1.0, 1),
            (0.0, 0),
        ]

        det_points = [
            [0.5, 0.0],
            [0.0, 0.0],
            [0.0, 0.5],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg)[0], det_points)

    def test_simple_bad_system(self):
        score_tg = [
            (1.0, 0),
            (0.0, 1),
        ]

        det_points = [
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg)[0], det_points)

    def test_multiple_points(self):
        score_tg = [
            (0.0, 0),
            (0.1, 1),
            (0.2, 0),
            (0.3, 1),
            (0.4, 1),
        ]

        det_points = [
            [0.6, 0.0],
            [0.4, 0.0],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.0, 0.2],
            [0.0, 0.4],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg)[0], det_points)


class ListSubsamplingTests(TestCase):
    def test_trivial(self):
        self.assertEqual(
            subsample_list([1, 2], 2),
            [1, 2]
        )

    def test_halving_longer(self):
        self.assertEqual(
            subsample_list([1, 2, 3, 4], 2),
            [1, 4]
        )

    def test_halving_uneven(self):
        self.assertEqual(
            subsample_list([1, 2, 3, 4, 5], 2),
            [1, 5]
        )

    def test_halving_uneven_2(self):
        self.assertEqual(
            subsample_list([1, 2, 3, 4, 5], 3),
            [1, 3, 5]
        )

    def test_every_third(self):
        self.assertEqual(
            subsample_list([1, 2, 3, 4, 5, 6], 2),
            [1, 6]
        )

    def test_include_most(self):
        self.assertIn(
            subsample_list([1, 2, 3, 4, 5, 6], 5),
            [
                [1, 3, 4, 5, 6],
                [1, 2, 4, 5, 6],
                [1, 2, 3, 5, 6],
                [1, 2, 3, 4, 6],
            ]
        )

    def test_include_4_of_6(self):
        self.assertIn(
            subsample_list([1, 2, 3, 4, 5, 6], 4),
            [
                [1] + middle + [6] for middle in [
                    [2, 3], [2, 4], [2, 5],
                    [3, 4], [3, 5],
                    [4, 5],
                ]
            ]
        )


class AreaComputationTests(TestCase):
    def test_trivial(self):
        self.assertAlmostEqual(area_under_curve([0.0, 1.0], [1.0, 0.0]), 0.5)

    def test_breakpoint(self):
        self.assertAlmostEqual(area_under_curve([0.0, 0.1, 1.0], [1.0, 0.1, 0.0]), 0.1*0.1 + 2*0.1*0.9/2)

    def test_end_addition(self):
        self.assertAlmostEqual(area_under_curve([0.1], [0.1]), 0.1*0.1 + 2*0.1*0.9/2)


class EerComputationTests(TestCase):
    def test_trivial(self):
        self.assertAlmostEqual(eer([0.0, 1.0], [1.0, 0.0]), 0.5)

    def test_skewed(self):
        self.assertAlmostEqual(eer([0.0, 1.0], [0.5, 0.0]), 1.0/3.0)

    def test_exact_hit(self):
        self.assertAlmostEqual(eer([0.0, 0.1, 1.0], [1.0, 0.1, 0.0]), 0.1)
