import unittest

import numpy as np

import matplotlib.pyplot as plt
import scipy.signal

from EMD.oned_filters import ButterworthLPF


def simple_sine_wave(freq, time):
    return np.sin(time * np.pi * freq)


if __name__ == '__main__':
    time_frame = np.linspace(0, 10, 1000)
    hf_sig = 0.1 * simple_sine_wave(10, time_frame)
    lf_sig = simple_sine_wave(1, time_frame)
    sig = lf_sig + hf_sig
    lpf = ButterworthLPF()
    plt.plot(time_frame, lf_sig, 'g')
    plt.plot(time_frame, sig, 'b')
    plt.plot(time_frame, lpf(sig), 'r')
    plt.show()
