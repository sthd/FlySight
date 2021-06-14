import unittest

import numpy as np

from EMD.dual_signal_processor import DualSignalProcessor


class TestDualSignalProcessor(unittest.TestCase):
    multiplier = DualSignalProcessor()

    def test_1x1_get_1(self):
        in1 = np.array([1])
        self.assertEqual(1, self.multiplier(in1, in1))

    def test_2x2_get_4(self):
        in2 = np.array([2])
        self.assertEqual(4, self.multiplier(in2, in2))

    def test_vector_multiplication(self):
        in_v = np.array([1, 2, 3, 4, 5])
        for expected, actual in zip([i*i for i in in_v], self.multiplier(in_v, in_v)):
            self.assertEqual(expected, actual)

    def test_subtraction(self):
        subtract = DualSignalProcessor(np.subtract)
        in_v = np.array([1, 2, 3, 4, 5])
        for expected, actual in zip([0, 1, 2, 3, 4], subtract(in_v, np.ones(5))):
            self.assertEqual(expected, actual)

    def test_different_lengths_in_input_shortest_will_be_padded_with_0(self):
        in_short = np.ones(10)
        in_long = np.ones(15)
        for i, val in enumerate(self.multiplier(in_short, in_long)):
            self.assertEqual(i < 10, val)
        for i, val in enumerate(self.multiplier(in_long, in_short)):
            self.assertEqual(i < 10, val)

    def test_complex_operation(self):
        def square_error(sig1, sig2):
            err = np.subtract(sig1, sig2)
            return np.multiply(err, err)

        se = DualSignalProcessor(square_error)

        in1 = 5 * np.ones(10)
        in2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_array = np.array([16, 9, 4, 1, 0, 1, 4, 9, 16, 25])

        for expected, actual in zip(expected_array, se(in1, in2)):
            self.assertEqual(expected, actual)

    def test_auxiliary_attributes(self):
        def sum_is_equal(sig1, sig2, val):
            return np.array([1 if v1+v2 == val else 0 for v1, v2 in zip(sig1, sig2)])

        in1 = np.array([2, 6, 8])
        in2 = np.array([4, 4, 2])

        sie = DualSignalProcessor(sum_is_equal, val=10)

        for i1, i2, v in zip(in1, in2, sie(in1, in2)):
            self.assertEqual(i1+i2 == 10, v)


if __name__ == '__main__':
    unittest.main()
