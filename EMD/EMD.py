import matplotlib.pyplot as plt
import numpy as np

from oned_filters import ButterworthLPF
from dual_signal_processor import DualSignalProcessor


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

test_EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)



def sanity_check(module):
    x_00 = np.array([0, 0])
    white = 2000
    x_01 = np.array([0, white])
    x_10 = np.array([white, 0])
    x_11 = np.array([white, white])

    print_sc_response(module, x_00, x_00)
    print_sc_response(module, x_11, x_11)
    print_sc_response(module, x_01, x_10)
    print_sc_response(module, x_10, x_01)

    print_sc_response(module, x_10, x_10)
    print_sc_response(module, x_01, x_01)

    print_sc_response(module, x_00, x_11)
    print_sc_response(module, x_11, x_00)


def print_sc_response(module, sig1, sig2):
    print(f"Costant Response of:\n"
          f"{sig1}\n"
          f"{sig2}\n"
          f"is {module(sig1, sig2)}")


# def emd_impulse_response():
#     impulse = np.zeros(length)
#     impulse_delay = np.zeros(length)
#     impulse[length // 2] = 1
#     impulse_delay[length // 2 + 1] = 1
#     plt.plot(time_axis, impulse, 'b')
#     plt.plot(time_axis, impulse_delay, 'r')
#     plt.show()
#     emd_response = EMD(impulse, impulse_delay)
#     print(emd_response)
#     plt.plot(time_axis, emd_response)
#     plt.show()
#
#
# def emd_rect_response():
#     rect = np.zeros(length)
#     rect_delay = np.zeros(length)
#     rect[length // 2: length // 2 + length // 10] = 1
#     d = 5
#     rect_delay[length // 2 + d: length // 2 + length // 10 + d] = 1
#     plt.plot(time_axis, rect, 'b')
#     plt.plot(time_axis, rect_delay, 'r')
#     plt.show()
#     emd_response = EMD(rect, rect_delay)
#     print(emd_response)
#     plt.plot(time_axis, emd_response)
#     plt.show()
#
#
# def sine_wave_response():
#     sin = np.sin(time_axis * np.pi / 60)
#     cos = np.cos(time_axis * np.pi / 60)
#     plt.plot(time_axis, sin, 'b')
#     plt.plot(time_axis, cos, 'r')
#     plt.show()
#     emd_response = EMD(sin, cos)
#     print(emd_response)
#     plt.plot(time_axis, emd_response)
#     plt.show()


if __name__ == '__main__':
    sanity_check(EMD)
    # length = 1000
    # time_axis = np.linspace(0, length, length)
    # emd_impulse_response()
    # emd_rect_response()
    # sine_wave_response()

    #in2 = np.array([0,1])
    #in1 = np.array([1,0])
    #results = test_EMD(in1, in2)
    #print("response is " )
    #print(results)