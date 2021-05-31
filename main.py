import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal

WHITE_LEVEL = 255

def greyscale_plot(image):
    plt.imshow(image / WHITE_LEVEL, cmap='gray')
    plt.show()


def make_gaussian_kernel(size: int, sigma: float = 1) -> np.array:
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    return np.exp(-((d - 0) ** 2 / (2.0 * sigma ** 2)))

ker = make_gaussian_kernel(16, .5)
greyscale_plot(ker)

img = cv2.imread('/Users/elior/PycharmProjects/FlySight/pic.jpg', cv2.IMREAD_GRAYSCALE)


greyscale_plot(img)

pr_sim = scipy.signal.convolve2d(img, ker)
greyscale_plot(pr_sim)
