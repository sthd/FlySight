import cv2
import scipy.signal

from auxFunctions import greyscale_plot, make_gaussian_kernel

if __name__ == '__main__':
    ker = make_gaussian_kernel(16, .5)
    greyscale_plot(ker)

    img = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

    greyscale_plot(img)

    pr_sim = scipy.signal.convolve2d(img, ker)
    greyscale_plot(pr_sim)
