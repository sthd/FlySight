from dataclasses import dataclass

import scipy.signal


@dataclass
class ButterworthLPF:
    sample_rate: float = 30.0
    cutoff: float = 1.1
    order: int = 6  # sin wave can be approx represented as polynomial

    def __post_init__(self):
        self.numerator, self.denom = scipy.signal.butter(self.order, 2 * self.cutoff / self.sample_rate, btype='low')

    def butter_lowpass_filter(self, data):
        return scipy.signal.filtfilt(self.numerator, self.denom, data)

    def __call__(self, input_signal):
        return self.butter_lowpass_filter(input_signal)
    
