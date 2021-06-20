import gif2numpy
import matplotlib.pyplot as plt
import numpy as np

import auxFunctions as aux

import cv2

from EMD.oned_filters import ButterworthLPF
from EMD.dual_signal_processor import DualSignalProcessor

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
    print(len(gframes))
    print(gframes[0].shape)
    print(gframes[0])
    gframes = np.array((gframes))
    print(gframes.shape)

    pr = PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), (500, 500), (500 ** 2)//1)
    pr_g_frames = pr.receive(gframes)
    # for frame in pr_g_frames:
    #     aux.greyscale_plot(frame)

    long_movie = np.array(pr_g_frames.repeat(10, axis=0))
    print(long_movie.shape)

    single_emd_response = EMD(long_movie[:, 100, 100], long_movie[:, 100, 101])

    plt.plot(long_movie[:, 100, 100], 'r')
    plt.plot(long_movie[:, 100, 101], 'b')
    plt.show()

    plt.plot(single_emd_response)
    plt.show()




    x = 42




