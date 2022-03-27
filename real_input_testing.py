import cv2
import scipy.signal

import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter

if __name__ == '__main__':
    # name = "Elior"
    # print(f"Hi {name}!!! :-)")
    ker = aux.make_gaussian_kernel(9, .5)
    # aux.greyscale_plot(ker)

    img = cv2.imread('data/Demoiselle_Crane.jpg', cv2.IMREAD_GRAYSCALE)

    # aux.greyscale_plot(img)
    print(img.shape)

    pr = PhotoreceptorImageConverter(ker, img.shape, 6000)
    res = pr.apply(img)
    aux.greyscale_plot(res)
