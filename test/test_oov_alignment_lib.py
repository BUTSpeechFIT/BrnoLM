from unittest import TestCase

from brnolm.oov_clustering.oov_alignment_lib import align, extract_mismatch
from brnolm.oov_clustering.oov_alignment_lib import find_in_mismatches, number_of_errors


class AlignTest(TestCase):
    def test_trivial(self):
        a = "a b".split()
        b = "a b".split()

        expected = [
            (['a'], ['a']),
            (['b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_element(self):
        a = "a".split()
        b = "a".split()

        expected = [
            (['a'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_substitution_only(self):
        a = "a".split()
        b = "b".split()

        expected = [
            (['a'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_insertion(self):
        a = "a".split()
        b = "a b".split()

        expected = [
            (['a'], ['a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_insertion(self):
        a = "a".split()
        b = "a b c".split()

        expected = [
            (['a'], ['a', 'b', 'c']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_insertion_reversed(self):
        a = "b".split()
        b = "a b".split()

        expected = [
            (['b'], ['a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_insertion_reversed(self):
        a = "b".split()
        b = "c a b".split()

        expected = [
            (['b'], ['c', 'a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_deletion(self):
        a = "a b".split()
        b = "a".split()

        expected = [
            (['a', 'b'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_deletion(self):
        a = "a b c".split()
        b = "a".split()

        expected = [
            (['a', 'b', 'c'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_deletion_reversed(self):
        a = "a b".split()
        b = "b".split()

        expected = [
            (['a', 'b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_deletion_reversed(self):
        a = "c a b".split()
        b = "b".split()

        expected = [
            (['c', 'a', 'b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_substituion(self):
        a = "a b c d".split()
        b = "a x y d".split()

        expected = [
            (['a'], ['a']),
            (['b'], ['x']),
            (['c'], ['y']),
            (['d'], ['d']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_inner_insertion(self):
        a = "a d".split()
        b = "a b c d".split()

        expected_1 = [
            (['a'], ['a', 'b', 'c']),
            (['d'], ['d']),
        ]
        expected_2 = [
            (['a'], ['a']),
            (['d'], ['b', 'c', 'd']),
        ]

        self.assertIn(align(a, b), [expected_1, expected_2])

    def test_inversion_at_end(self):
        a = "a b c d".split()
        b = "a b d c".split()

        expected = [
            (['a'], ['a']),
            (['b'], ['b']),
            (['c'], ['d']),
            (['d'], ['c']),
        ]

        self.assertEqual(align(a, b), expected)


class MismatchExtractionTest(TestCase):
    def test_trivial(self):
        ali = [
            (['a'], ['a'])
        ]
        expectation = [
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_single_substitution(self):
        ali = [
            (['a'], ['b'])
        ]
        expectation = [
            (['a'], ['b'])
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_double_substitution(self):
        ali = [
            (['a'], ['a']),
            (['a'], ['b']),
            (['a'], ['b']),
            (['a'], ['a']),
            (['a'], ['b']),
        ]
        expectation = [
            (['a', 'a'], ['b', 'b']),
            (['a'], ['b'])
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_substitution_with_insertion(self):
        ali = [
            (['begin'], ['begin']),
            (['c'], ['a', 'b']),
            (['end'], ['end']),
        ]
        expectation = [
            (['c'], ['a', 'b']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_insertion_only(self):
        ali = [
            (['begin'], ['begin']),
            (['a'], ['a', 'b']),
            (['end'], ['end']),
        ]
        expectation = [
            ([], ['b']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_insertion_only_reversed(self):
        ali = [
            (['begin'], ['begin']),
            (['b'], ['a', 'b']),
            (['end'], ['end']),
        ]
        expectation = [
            ([], ['a']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_substitution_with_deletion(self):
        ali = [
            (['begin'], ['begin']),
            (['a', 'b'], ['c']),
            (['end'], ['end']),
        ]
        expectation = [
            (['a', 'b'], ['c']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_deletion_only(self):
        ali = [
            (['begin'], ['begin']),
            (['a', 'b'], ['a']),
            (['end'], ['end']),
        ]
        expectation = [
            (['b'], []),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_deletion_only_reversed(self):
        ali = [
            (['begin'], ['begin']),
            (['a', 'b'], ['b']),
            (['end'], ['end']),
        ]
        expectation = [
            (['a'], []),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_deletion_after_substitution(self):
        ali = [
            (['begin'], ['begin']),
            (['x'], ['y']),
            (['a', 'b'], ['b']),
            (['end'], ['end']),
        ]
        expectation = [
            (['x', 'a'], ['y']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)


class FindingInMismatchesTest(TestCase):
    def test_trivial(self):
        mismatches = [
            (['x'], ['a'])
        ]
        expectation = (['x'], ['a'])

        self.assertEqual(find_in_mismatches(mismatches, 'x'), expectation)

    def test_finding(self):
        mismatches = [
            (['c'], ['d']),
            (['x'], ['a']),
            (['b'], ['f']),
        ]
        expectation = (['x'], ['a'])

        self.assertEqual(find_in_mismatches(mismatches, 'x'), expectation)

    def test_multiple_elements_in_mismatch(self):
        mismatches = [
            (['c'], ['d']),
            (['c', 'b', 'x'], ['a']),
            (['b'], ['f']),
        ]
        expectation = (['c', 'b', 'x'], ['a'])

        self.assertEqual(find_in_mismatches(mismatches, 'x'), expectation)


class NumberOfErrorsTest(TestCase):
    def test_trivial(self):
        mismatches = [
            (['x'], ['a'])
        ]

        self.assertEqual(number_of_errors(mismatches), 1)

    def test_no_mismatch(self):
        self.assertEqual(number_of_errors([]), 0)

    def test_multiple_substitutions(self):
        mismatches = [
            (['x'], ['a']),
            (['x'], ['a'])
        ]

        self.assertEqual(number_of_errors(mismatches), 2)

    def test_single_insertion(self):
        mismatches = [
            (['x'], ['a', 'b']),
        ]

        self.assertEqual(number_of_errors(mismatches), 2)

    def test_single_deletion(self):
        mismatches = [
            (['a', 'b'], ['x']),
        ]

        self.assertEqual(number_of_errors(mismatches), 2)
