import numpy as np

import SP.signal_processing as spb
from EMD.lpf_plus_EMD import butter_lowpass_filter


# from oned_filters import ButterworthLPF
# from dual_signal_processor import DualSignalProcessor


def emd_action(sig1, sig2, lpf, mul, sub):
    sig1_lpf = lpf(sig1)
    sig2_lpf = lpf(sig2)
    cross1 = mul(sig1_lpf, sig2)
    cross2 = mul(sig1, sig2_lpf)
    return sub(cross1, cross2)


# LPF = ButterworthLPF()
# MUL = DualSignalProcessor(np.multiply)
# SUB = DualSignalProcessor(np.subtract)
#
# EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)
#
# test_EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)


def make_basic_emd():
    cutoff = 3.667
    order = 6
    fs = 30  # sampling rate in HZ
    lpf = spb.SP_Block(butter_lowpass_filter, cutoff_frequency=cutoff, fs=fs, order=order)
    multiplier = spb.SP_Block(np.multiply, inputs=2)
    subtractor = spb.SP_Block(np.subtract, inputs=2)

    splitter = spb.SP_Split(2)
    frequency_split = splitter.send_to((lpf,))
    mul_prep = spb.SP_Arrange((0, 3, 1, 2))
    to_multiply = mul_prep.receive_from((frequency_split, frequency_split))
    to_sub = to_multiply.send_to((multiplier, multiplier))

    return subtractor.receive_from(to_sub)

