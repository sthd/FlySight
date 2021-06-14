import unittest

import numpy as np

from EMD.oned_filters import ButterworthLPF


class TestButterworthLPF(unittest.TestCase):
    def test_basic_filtering(self):
        time = np.linspace(0, 1e4)
        highf = np.pi * 100
        hf_sig = np.sin(time * highf)
        lowf = np.pi * 10
        lf_sig = np.sin(time * lowf)
        sig = lf_sig + hf_sig
        lpf = ButterworthLPF()
        for expected, actual in zip(lf_sig, lpf(sig)):
            self.assertAlmostEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
