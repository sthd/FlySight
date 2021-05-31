import unittest

import numpy as np

from flyConvol import PhotoreceptorImageConverter


class TestConv(unittest.TestCase):
    def test_init(self):
        pr = PhotoreceptorImageConverter(np.ones((3, 3)), (6, 8), 48)
        self.assertEqual(8, pr.horizontal)
        self.assertEqual(6, pr.vertical)
        self.assertEqual(1, pr.h_stride)
        self.assertEqual(1, pr.v_stride)
        self.assertEqual(1, pr.h_padding)
        self.assertEqual(1, pr.v_padding)

    def test_different_numbers(self):
        pr = PhotoreceptorImageConverter(np.ones((1, 1)), (2, 3), 7)
        self.assertEqual(3, pr.horizontal)
        self.assertEqual(2, pr.vertical)
        self.assertEqual(1, pr.h_stride)
        self.assertEqual(1, pr.v_stride)
        self.assertEqual(0, pr.h_padding)
        self.assertEqual(0, pr.v_padding)

    def test_one_photoreceptor(self):
        pr = PhotoreceptorImageConverter(np.ones((3, 3)), (3, 3), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(0, pr.h_padding)
        self.assertEqual(0, pr.v_padding)

    def test_round_to_zero(self):
        pr = PhotoreceptorImageConverter(np.ones((5, 5)), (5, 1), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(2, pr.h_padding)
        self.assertEqual(0, pr.v_padding)

    def test_negative_padding(self):
        pr = PhotoreceptorImageConverter(np.ones((1, 1)), (3, 3), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(-1, pr.h_padding)
        self.assertEqual(-1, pr.v_padding)


    def test_stride_2(self):
        pr = PhotoreceptorImageConverter(np.ones((3, 3)), (3, 4), 6)
        self.assertEqual(3, pr.horizontal)
        self.assertEqual(2, pr.vertical)
        self.assertEqual(1, pr.h_stride)
        self.assertEqual(2, pr.v_stride)
        self.assertEqual(1, pr.h_padding)
        self.assertEqual(1, pr.v_padding)

    def test_stride_2_neg_pad(self):
        pr = PhotoreceptorImageConverter(np.ones((1, 1)), (3, 4), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(-1, pr.h_padding)
        self.assertEqual(-1, pr.v_padding)

    def test_simple_application(self):
        pr = PhotoreceptorImageConverter(np.array([[1]]), (1, 1), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(0, pr.h_padding)
        self.assertEqual(0, pr.v_padding)
        pic = np.array([[1]])
        self.assertEqual(1, pr.apply(pic))


if __name__ == '__main__':
    unittest.main()
