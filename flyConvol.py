import numpy


class PhotoreceptorImageConverter:
    def __init__(self, kernel_size: int, rows: int, columns: int, photoreceptor_num: int):
        self.horizontal = numpy.sqrt((columns * photoreceptor_num) / rows)
        self.vertical = (photoreceptor_num / self.horizontal)
        self.h_stride = (columns / self.horizontal)
        self.v_stride = (rows / self.vertical)
        self.h_padding = ((kernel_size - self.h_stride) / 2)
        self.v_padding = ((kernel_size - self.v_stride) / 2)
