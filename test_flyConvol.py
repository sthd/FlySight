import unittest

import numpy as np

from flyConvol import PhotoreceptorImageConverter


class TestConv(unittest.TestCase):
    def assertEqualArrays(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        for e, a, in zip(np.nditer(expected), np.nditer(actual)):
            self.assertEqual(e, a)

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

    def test_simple_application_different_value(self):
        pr = PhotoreceptorImageConverter(np.array([[2]]), (1, 1), 1)
        self.assertEqual(1, pr.horizontal)
        self.assertEqual(1, pr.vertical)
        self.assertEqual(0, pr.h_padding)
        self.assertEqual(0, pr.v_padding)
        pic = np.array([[1]])
        self.assertEqual(2, pr.apply(pic))

    def test_application_kernel_1x1_pic_2x2(self):
        pic = np.ones((2, 2))
        pr = PhotoreceptorImageConverter(np.array([[2]]), pic.shape, 4)
        self.assertEqualArrays(2*np.ones((2, 2)), pr.apply(pic))

    def test_kernel_3x3_pic_2x2_zero_padding(self):
        pic = np.ones((2, 2))
        pr = PhotoreceptorImageConverter(np.ones((3, 3)), pic.shape, 4)
        self.assertEqualArrays(4 * np.ones((2, 2)), pr.apply(pic))

    def test_negative_padding_apply(self):
        pic = np.ones((3, 3))
        pr = PhotoreceptorImageConverter(np.array([[2]]), pic.shape, 1)
        self.assertEqualArrays(np.array([[2]]), pr.apply(pic))

    def test_negative_padding_4x4(self):
        pic = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])
        pr = PhotoreceptorImageConverter(np.array([[1]]), pic.shape, 1)
        self.assertEqualArrays(np.array([[1]]), pr.apply(pic))

    def test_apply_stride_2_kernel_1x1(self):
        pic = np.array([[1, 0, 2],
                        [0, 0, 0],
                        [3, 0, 4]])
        pr = PhotoreceptorImageConverter(np.array([[2]]), pic.shape, 4)
        self.assertEqualArrays(np.array([[2, 4],
                                         [6, 8]]), pr.apply(pic))


if __name__ == '__main__':
    unittest.main()
