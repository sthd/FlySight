import gif2numpy
import matplotlib.pyplot as plt
import numpy as np

import auxFunctions as aux

# import cv2

from EMD.oned_filters import ButterworthLPF
# from EMD.dual_signal_processor import DualSignalProcessor

BUFFER_SIZE = 120


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
GIF2 = "complex_stripes.gif"


def greyscale_gif(gif_path):
    frames, exts, image_specs = gif2numpy.convert(gif_path, False)
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]


def emd_row(buf, row_index):
    some_row = np.array(buf)[:, row_index, :]
    return [EMD(some_row[:, i], some_row[:, i + 1]) for i in range(some_row.shape[1] - 1)]


def angle_response_from_frequency_response_array(fr_array):
    angle_response_emd = []
    for fr in fr_array:
        integrand = []
        for idx, val in enumerate(fr):
            if idx:
                normalizer = idx ** -2
            else:
                normalizer = 0
            integrand.append(normalizer * val)
        angle_response_emd.append(sum(integrand))
    return angle_response_emd


if __name__ == '__main__':
    g_frames = greyscale_gif(GIF2)
    g_frames_t = [np.transpose(f) for f in g_frames]
    # for f in g_frames_t:
    #     aux.greyscale_plot(f)

    frame_area = g_frames[0].shape[0] * g_frames[0].shape[1]
    pr = PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), g_frames[0].shape, frame_area // 16)

    for buffer in pr.stream(g_frames_t * BUFFER_SIZE, buffer_size=BUFFER_SIZE):
        emd = emd_row(buffer, buffer[0].shape[0] // 2)
        if len(buffer) == BUFFER_SIZE:
            aux.greyscale_plot(np.array(emd))
            plt.plot(emd[10])
            plt.show()

        fr_emd = [np.abs(np.fft.rfft(tr)) for tr in emd]

        if len(buffer) == BUFFER_SIZE:
            aux.greyscale_plot(np.array(fr_emd))
            plt.plot(fr_emd[10])
            plt.show()

        ar_emd = angle_response_from_frequency_response_array(fr_emd)
        if len(buffer) == BUFFER_SIZE:
        #
        #     a_file = open("EMD_Response.txt", "w")
        #
        #     for row in ar_emd:
        #         np.savetxt(a_file, row)
        #

        #     a_file.close()
            #print(ar_emd.shape)
            plt.plot(ar_emd)
            plt.grid()
            plt.ylabel("EMD Response")
            plt.show()
