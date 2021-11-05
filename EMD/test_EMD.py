# Sanity check for EMD

import unittest

import numpy as np

import matplotlib.pyplot as plt
import scipy.signal

from EMD.EMD import EMD

class TestEMD(unittest.TestCase):
    EMD = DualSignalProcessor()
    #EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)

    def test_object_move_right_get_1(self):
        in1 = np.array([255, 0])
        in2 = np.array([0, 255])
        self.EMD(1, self.multiplier(in1, in1))

    def test_object_move_left_get_negative1(self):
        in1 = np.array([1])
        self.assertEqual(1, self.multiplier(in1, in1))

    def test_object_no_change_white_get_0(self):
        in1 = np.array([1])
        self.assertEqual(1, self.multiplier(in1, in1))

    def test_object_no_change_black_get_0(self):
        in1 = np.array([1])
        self.assertEqual(1, self.multiplier(in1, in1))


if __name__ == '__main__':
    unittest.main()

