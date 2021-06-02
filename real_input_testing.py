import cv2
import scipy.signal

import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter

if __name__ == '__main__':
    ker = aux.make_gaussian_kernel(17, .5)
    # aux.greyscale_plot(ker)

    img = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

    # aux.greyscale_plot(img)
    print(img.shape)

    pr = PhotoreceptorImageConverter(ker, img.shape, 65000)
    res = pr.apply(img)
    aux.greyscale_plot(res)
