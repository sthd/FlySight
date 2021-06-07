import gif2numpy

import auxFunctions as aux
GIF = "stripes.gif"

if __name__ == '__main__':
    frames, exts, image_specs = gif2numpy.convert(GIF, "greyscale")
    print(len(frames))
    for frame in frames:
        aux.greyscale_plot(frame)
    x=42




