import unittest

from flyConvol import PhotoreceptorImageConverter


class TestConv(unittest.TestCase):
    def test_init(self):
        pr = PhotoreceptorImageConverter(3, 6, 8, 48)
        self.assertEqual(8, pr.horizontal)
        self.assertEqual(6, pr.vertical)
        self.assertEqual(1, pr.h_stride)
        self.assertEqual(1, pr.h_padding)
        self.assertEqual(1, pr.v_padding)




if __name__ == '__main__':
    unittest.main()
