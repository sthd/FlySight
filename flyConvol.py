import math
import numpy
import numpy as np


class PhotoreceptorImageConverter:
    def __init__(self, kernel: np.array, pic_shape: (int, int), photoreceptor_num: int):
        # Assumption: kernel_size is an odd number
        self.kernel = kernel
        self.horizontal: int = numpy.sqrt((pic_shape[1] * photoreceptor_num) / pic_shape[0]).round().astype(int)
        if self.horizontal < 1:
            self.horizontal = 1
        self.vertical = round(photoreceptor_num / self.horizontal)
        self.h_stride = round(pic_shape[1] / self.horizontal)
        self.v_stride = round(pic_shape[0] / self.vertical)
        self.h_padding = math.ceil((self.kernel.shape[0] - self.h_stride) / 2)
        self.v_padding = math.ceil((self.kernel.shape[1] - self.v_stride) / 2)

    def apply(self, pic):
        if self.h_padding < 0 or self.v_padding < 0:
            padded_pic = self._slice_pic(pic)
        else:
            padded_pic = self._make_zero_padded_input(pic)
        output = np.zeros((self.vertical, self.horizontal))
        half_ker = (self.kernel.shape[0]-1)//2
        self._convolve(half_ker, output, padded_pic)
        return output

    def _slice_pic(self, pic):
        padded_pic = pic
        if self.v_padding < 0:
            padded_pic = padded_pic[-self.v_padding:self.v_padding, :]
            self.v_padding = 0
        if self.h_padding < 0:
            padded_pic = padded_pic[:, -self.h_padding:self.h_padding]
            self.h_padding = 0
        return padded_pic

    def _convolve(self, half_ker, output, padded_pic):
        for i in range(self.vertical):
            for j in range(self.horizontal):
                input_row = i * self.v_stride + self.v_padding
                input_column = j * self.h_stride + self.h_padding
                output[i, j] = self._overlay_kernel(half_ker, input_row, input_column, padded_pic)

    def _overlay_kernel(self, half_ker, i, j, padded_pic):
        return np.multiply(self.kernel, self._kernel_sized_portion(padded_pic, half_ker, i, j)).sum()

    @staticmethod
    def _kernel_sized_portion(pic, half_ker_size, center_row, center_col):
        return pic[center_row - half_ker_size:center_row + half_ker_size + 1,
                   center_col - half_ker_size:center_col + half_ker_size + 1]

    def _make_zero_padded_input(self, pic: np.array) -> np.array:
        padded_pic = np.zeros((pic.shape[0] + 2 * self.v_padding, pic.shape[1] + 2 * self.h_padding))
        for i in range(self.v_padding, padded_pic.shape[0] - self.v_padding):
            for j in range(self.h_padding, padded_pic.shape[1] - self.h_padding):
                padded_pic[i, j] = pic[i - self.v_padding, j - self.h_padding]
        return padded_pic
