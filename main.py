import cv2
import scipy.signal

import auxFunctions as aux

if __name__ == '__main__':
    ker = aux.make_gaussian_kernel(17, .5)
    aux.greyscale_plot(ker)

    img = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

    aux.greyscale_plot(img)

    pr_sim = scipy.signal.convolve2d(img, ker)
    aux.greyscale_plot(pr_sim)


