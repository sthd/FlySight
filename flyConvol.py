import math

import numpy
import numpy as np


class PhotoreceptorImageConverter:
    def __init__(self, kernel: np.array, pic_shape: (int, int), photoreceptor_num: int):
        # Assumption: kernel_size is an odd number
        self.horizontal = numpy.sqrt((pic_shape[1] * photoreceptor_num) / pic_shape[0]).round()
        if self.horizontal < 1:
            self.horizontal = 1
        self.vertical = round(photoreceptor_num / self.horizontal)
        self.h_stride = round(pic_shape[1] / self.horizontal)
        self.v_stride = round(pic_shape[0] / self.vertical)
        self.h_padding = math.ceil((kernel.shape[0] - self.h_stride) / 2)
        self.v_padding = math.ceil((kernel.shape[1] - self.v_stride) / 2)

    def apply(self, pic):
        return 1


