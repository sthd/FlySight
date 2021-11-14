import numpy as np


class DualSignalProcessor:
    """Old unused version of generic model."""
    def __init__(self, func=np.multiply, **kwargs):
        self.function = func
        self.aux = kwargs

    def __call__(self, sig1, sig2):
        sig1 = self.pad_signal_if_needed(sig1, sig2)
        sig2 = self.pad_signal_if_needed(sig2, sig1)
        return self.function(sig1, sig2, **self.aux)

    @staticmethod
    def pad_signal_if_needed(sig1, sig2):
        return np.pad(sig1, (0, (len(sig2) - len(sig1)) * (len(sig1) < len(sig2))))
