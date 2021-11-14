from dataclasses import dataclass

import numpy as np
import scipy.signal


@dataclass
class ButterworthLPF:
    """Old unused model."""
    sample_rate: float = 30.0
    cutoff: float = 1.1
    order: int = 6  # sin wave can be approx represented as polynomial

    def __post_init__(self):
        self.numerator, self.denom = scipy.signal.butter(self.order, 2 * self.cutoff / self.sample_rate, btype='low')

    def butter_lowpass_filter(self, data: np.array):
        try:
            return scipy.signal.filtfilt(self.numerator, self.denom, data)
        except ValueError:
            return scipy.signal.filtfilt(self.numerator, self.denom, np.pad(data, (0, self.numerator.shape[0] * self.denom.shape[0])))

    def __call__(self, input_signal):
        return self.butter_lowpass_filter(input_signal)
    
