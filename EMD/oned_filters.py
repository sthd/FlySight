import numpy as np


class ButterworthLPF:
    def __call__(self, *args, **kwargs):
        return np.sin(np.linspace(0, 1e4) * np.pi * 10)
    
