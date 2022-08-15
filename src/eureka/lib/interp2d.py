import numpy as np


def interp2d(image, expand=5, y=None, x=None, yi=None, xi=None):
    """
    This function oversamples a 2D frame (image) which can be used if the user decides
    that the resolution of the image is not enough and they want to split a pixel
    into more subpixels.

    Parameters
    ----------
    image : 2D numpy array
        Contains the 2D frame which will be oversampled in pixels
    expand : int
        The factor by which a pixel should be oversampled. If set to 5, a pixel will turn into 25 subpixels.
    y : 1D numpy array
        np.arange(ny), with ny being the number of pixels in the y direction
    x : 1D numpy array
        np.arange(nx), with ny being the number of pixels in the x direction
    yi : 1D numpy array
        np.linspace(0, ny - 1, isz[0]),
        with isz = np.array(sz, dtype=int) + (np.array(sz, dtype=int) - 1) * (expand - 1)
        and sz = np.shape(image)
    xi : 1D numpy array
        np.linspace(0, ny - 1, isz[0]),
        with isz = np.array(sz, dtype=int) + (np.array(sz, dtype=int) - 1) * (expand - 1)
        and sz = np.shape(image)
    """
    sz = np.shape(image)
    imagen = np.zeros(expand * (np.array(sz) - 1) + 1)
    szi = np.shape(imagen)

    if y is None:
        y = np.arange(sz[0])
        x = np.arange(sz[1])
        yi = np.linspace(0, sz[0]-1, szi[0])
        xi = np.linspace(0, sz[1]-1, szi[1])

    for k in np.arange(sz[0]):
        imagen[k] = np.interp(xi, x, image[k])

    for k in np.arange(szi[1]):
        imagen[:, k] = np.interp(yi, y, imagen[0:sz[0], k])

    return imagen
