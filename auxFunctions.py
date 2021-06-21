import numpy as np
from matplotlib import pyplot as plt

WHITE_LEVEL = 255


def greyscale_plot(image):
    plt.imshow(image / WHITE_LEVEL, cmap='gray')
    plt.show()


def make_gaussian_kernel(size: int, sigma: float = 1) -> np.array:
    coefficient = 1/((2 * np.pi) * (sigma ** 2))
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    gaus = np.exp(-((d - 0) ** 2 / (2.0 * sigma ** 2)))
    return gaus / gaus.sum()
