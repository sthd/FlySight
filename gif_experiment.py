import gif2numpy
import matplotlib.pyplot as plt
import numpy as np

import auxFunctions as aux

import cv2

from EMD.oned_filters import ButterworthLPF
from EMD.dual_signal_processor import DualSignalProcessor

BUFFER_SIZE = 100


def emd_action(sig1, sig2, lpf, mul, sub):
    sig1_lpf = lpf(sig1)
    sig2_lpf = lpf(sig2)
    cross1 = mul(sig1_lpf, sig2)
    cross2 = mul(sig1, sig2_lpf)
    return sub(cross1, cross2)

LPF = ButterworthLPF()
MUL = DualSignalProcessor(np.multiply)
SUB = DualSignalProcessor(np.subtract)

EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)


from flyConvol import PhotoreceptorImageConverter

GIF = "stripes.gif"

if __name__ == '__main__':
    frames, exts, image_specs = gif2numpy.convert(GIF, False)
    gframes = []
    for frame in frames:
        gframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    pr = PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), (500, 500), (500 ** 2)//64)

    for buffer in pr.stream(gframes * BUFFER_SIZE, buffer_size=BUFFER_SIZE):
        npa_buf = np.array(buffer)
        some_row = npa_buf[:, 50, :]
        print(some_row.shape)
        emd = []
        for i in range(some_row.shape[1] - 1):
            emd.append(EMD(some_row[:, i], some_row[:, i + 1]))
        # if len(buffer) == BUFFER_SIZE:
        #     aux.greyscale_plot(np.array(emd))



    x = 42




