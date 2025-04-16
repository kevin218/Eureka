import numpy as np
from matplotlib.path import Path


def disk(r, ctr, size, status=False):
    """
    This function returns a boolean array containing a disk.

    The disk is centered at (ctr[0], ctr[1]), and has radius r. The array is
    (size[0], size[1]) in size and has boolean type. Pixel values of False
    indicate that the center of a pixel is within r of (ctr[0], ctr[1]). Pixel
    values of True indicate the opposite. The center of each pixel is the
    integer position of that pixel.

    Parameters
    ----------
    r : float
        The radius of the disk.
    ctr : tuple, list, or array
        The x,y position of the center of the disk, 2-element vector.
    size : tuple, list, or array
        The x,y size of the output array, 2-element vector.
    status : bool; optional
        If True, return the status optional output.

    Returns
    -------
    ret : ndarray
        A boolean array where True if outside the disk or False if inside
        the disk.
    retstatus : int; optional
        Set to 1 if any part of the disk is outside the image boundaries.
        Only returned if status==True.

    """
    # check if disk is off image
    retstatus = int(ctr[0] - r < 0 or ctr[0] + r > size[0]-1 or
                    ctr[1] - r < 0 or ctr[1] + r > size[1]-1)

    # calculate pixel distance from center
    # print('disk size:',  size)
    ind = np.indices(size)
    fdisk = (ind[0]-ctr[0])**2.0 + (ind[1]-ctr[1])**2.0

    # return mask disk (and status if requested)
    ret = fdisk > r**2.0
    if status:
        ret = ret, retstatus
    return ret


def hex(r, ctr, size, status=False):
    """
    This function returns a byte array containing a hexagon.

    The hexagon is centered at (ctr[0], ctr[1]), and is circumscribed by a
    circle of radius r. The array is (size[0], size[1]) in size and has byte
    type. Pixel values of 1 indicate that the center of a pixel is within the
    hexagonal aperture. Pixel values of 0 indicate the opposite. The center
    of each pixel is the integer position of that pixel.

    Parameters
    ----------
    r : float
        The radius of the circle circumscribing the hexagon.
    ctr : tuple, list, or array
        The x,y position of the center of the hexagon, 2-element vector.
    size : tuple, list, or array
        The x,y size of the output array, 2-element vector.
    status : bool; optional
        If True, return the status optional output.

    Returns
    -------
    ret : ndarray
        A boolean array where False if outside the hexagon or True if inside
        the hexagon.
    retstatus : int; optional
        Set to 1 if any part of the hexagon is outside the image boundaries.
        Only returned if status==True.

    """

    # check if hex is off image (same check as disk, for now)
    retstatus = int(ctr[0] - r < 0 or ctr[0] + r > size[0]-1 or
                    ctr[1] - r < 0 or ctr[1] + r > size[1]-1)

    # make hexagon, oriented like the JWST mirrors (corner vertex at 12:00)
    yvert = ctr[1] + r*np.cos(2*np.pi*np.arange(6)/6)
    xvert = ctr[0] - r*np.sin(2*np.pi*np.arange(6)/6)
    hexverts = np.vstack((xvert, yvert)).T

    # use matplotlib Path to make hexagon
    poly_path = Path(hexverts)

    # get list of coordinates for each pixel in image
    ind = np.indices(size)
    coors = np.hstack((ind[0].reshape(-1, 1), ind[1].reshape(-1, 1)))

    # use matplotlib.path to do easy masking
    ret = ~poly_path.contains_points(coors).reshape(size[0], size[1])

    # return mask disk (and status if requested)
    if status:
        ret = ret, retstatus
    return ret