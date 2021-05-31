import math

import numpy


class PhotoreceptorImageConverter:
    def __init__(self, kernel_size: int, rows: int, columns: int, photoreceptor_num: int):
        # Assumption: kernel_size is an odd number
        self.horizontal = numpy.sqrt((columns * photoreceptor_num) / rows).round()
        if self.horizontal < 1:
            self.horizontal = 1
        self.vertical = round(photoreceptor_num / self.horizontal)
        self.h_stride = round(columns / self.horizontal)
        self.v_stride = round(rows / self.vertical)
        self.h_padding = math.ceil((kernel_size - self.h_stride) / 2)
        self.v_padding = math.ceil((kernel_size - self.v_stride) / 2)

