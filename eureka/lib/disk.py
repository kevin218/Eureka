import numpy as np


def disk(r, ctr, size, status=False):
    """This function returns a byte array containing a disk.

    The disk is centered at (ctr[0], ctr[1]), and has radius r. The array is
    (size[0], size[1]) in size and has byte type. Pixel values of 1 indicate
    that the center of a pixel is within r of (ctr[0], ctr[1]). Pixel values
    of 0 indicate the opposite. The center of each pixel is the integer
    position of that pixel.

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
        A boolean array where False if outside the disk or True if inside
        the disk.
    retstatus : int; optional
        Set to 1 if any part of the disk is outside the image boundaries.
        Only returned if status==True.

    Notes
    -----
    History:

    - 2003 April 4; Joseph Harrington, jh@oobleck.astro.cornell.edu
        Initial version.
    - 2004 Feb 27; jh
        Added alternate input method
    - 2005 Nov 16; jh
        Added STATUS, simplified disk calculation,
        use double precision
    """
    # check if disk is off image
    retstatus = int(ctr[0] - r < 0 or ctr[0] + r > size[0]-1 or
                    ctr[1] - r < 0 or ctr[1] + r > size[1]-1)

    # calculate pixel distance from center
    # print('disk size:',  size)
    ind = np.indices(size)
    fdisk = (ind[0]-ctr[0])**2.0 + (ind[1]-ctr[1])**2.0

    # return mask disk (and status if requested)
    ret = fdisk <= r**2.0
    if status:
        ret = ret, retstatus
    return ret
