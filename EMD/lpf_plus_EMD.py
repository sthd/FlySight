import numpy as np
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
import SP.signal_processing as spb


# Receives a signal in Time Domain,  desired cutoff frequency, sampling rate, order is set as 5
# Returns the signal in Time Domain after filtering
def butter_lowpass_filter(signal, cutoff_frequency, fs, order=5):
    nyq = 0.5 * fs
    normalised_cutoff = cutoff_frequency / nyq
    # butter returns Numerator (b) and Denominator (a) polynomials of the IIR filter
    b, a = butter(order, normalised_cutoff, btype='low', analog=False) #Butterworth filter design
    filtered_signal = lfilter(b, a, signal)  #Filter data along one-dimension with the designed IIR filter
    return filtered_signal

def lpf(signal):
    return butter_lowpass_filter(signal, cutoff, fs, order)

def perform_EMD(signal1, signal2):
    lpf1 = lpf(signal1)
    lpf2 = lpf(signal2)
    cross1 = np.multiply(lpf1, signal2)
    cross2 = np.multiply(signal1, lpf2)
    EMD_response = np.subtract(cross1, cross2)
    return EMD_response


if __name__ == '__main__':
    # Filter parameters
    cutoff = 3.667
    order = 6
    fs = 30  # sampling rate in HZ
    # fs = 32        #### in order to try array of size 2

    # Signals' parameters
    T = 0.5  # interval in seconds
    # T= 0.0625         ### in order to try array of size 2
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)

    #The lpf recovers the 1.2 Hz signal

    # expanding right
    # signal1 = np.array([0, 0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0])
    # signal2 = np.array([0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0])

    # big object moving right
    # signal1 = np.array([0, 0, 0, 0, 12, 20, 20, 20, 30, 0, 0, 0, 0, 0, 0])
    # signal2 = np.array([0, 0, 0, 0, 0, 0, 0, 12, 20, 20, 30, 0, 0, 0, 0])

    # 3 seperate objects moving right
    signal1 = np.array([0, 0, 0, 0, 0, 20, 0, 20, 0, 20, 0, 0, 0, 0, 0])
    signal2 = np.array([0, 0, 0, 0, 0, 0, 20, 0, 20, 0, 20, 0, 0, 0, 0])

    # signal1 = np.array([0, 20])
    # signal2 = np.array([20, 0])

    EMD_response = perform_EMD(signal1, signal2)  # executing EMD on 1D signals

    plt.subplot(3, 1, 1)
    plt.plot(t, signal1, 'b-', label='signal1')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, signal2, 'b-', label='signal2')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, EMD_response, 'g-', linewidth=2, label='EMD Response')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.8)
    plt.show()
