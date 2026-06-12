import numpy as np


def trimimage(data, c, r, mask=None, uncd=None):
    """
    Extracts a rectangular area of an image masking out of bound pixels.

    Parameters
    ----------
    data  : 2D ndarray
        Image from where extract a sub image.
    c: 2-elements tuple
        yc, xc. Position in data, where the extracted image will be centered.
        yc and xc must have integer values.
    r : 2-elements tuple
        yr, xr. Semi-length of the extracted image. Integer values.
    mask : 2D ndarray
        If specified, this routine will extract the mask subimage
        as well. out of bound pixels values will have value oob.
    uncd : 2D ndarray
        If specified, this routine will extract the uncd subimage
        as well.

    Returns
    -------
    Tuple, up to 3 2D-ndarray images containing the extracted image,
    its mask (if specified) and a uncd array (if specified). The shape
    of each array is (2*yr+1, 2*xr+1).

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> from imageedit import *
        >>> import numpy as np

        >>> # Create  a data image and its mask
        >>> data  = np.arange(25).reshape(5,5)
        >>> print(data)
        [[ 0  1  2  3  4]
        [ 5  6  7  8  9]
        [10 11 12 13 14]
        [15 16 17 18 19]
        [20 21 22 23 24]]
        >>> msk = np.zeros(np.shape(data), dtype=bool)
        >>> msk[1:4,2] = True

        >>> # Extract a subimage centered on (3,1) of shape (3,5)
        >>> dyc,dxc = 3,1
        >>> subim, mask = trimimage(data, (dyc,dxc), (1,2), mask=msk)
        >>> print(subim)
        [[  0.  10.  11.  12.  13.]
        [  0.  15.  16.  17.  18.]
        [  0.  20.  21.  22.  23.]]
        >>> print(mask)
        [[ True  False  False  True  False]
        [ True  False  False  True  False]
        [ True  False  False  False  False]]
    """
    (yc, xc) = c
    (yr, xr) = r

    # Shape of original data
    ny, nx = np.shape(data)

    # The extracted image and mask
    im = np.zeros((2 * int(yr) + 1, 2 * int(xr) + 1))

    # coordinates of the limits of the extracted image
    uplim = int(yc + yr + 1)  # upper limit
    lolim = int(yc - yr)  # lower limit
    rilim = int(xc + xr + 1)  # right limit
    lelim = int(xc - xr)  # left  limit

    # Ranges (in the original image):
    bot = np.amax((0, lolim))  # bottom
    top = np.amin((ny, uplim))  # top
    lft = np.amax((0, lelim))  # left
    rgt = np.amin((nx, rilim))  # right

    im[bot - lolim:top - lolim, lft - lelim:rgt - lelim] = data[bot:top,
                                                                lft:rgt]
    ret = (im, )

    if mask is not None:
        # The mask is initialized to True to mask out-of-bounds pixels
        ma = np.ones((2 * int(yr) + 1, 2 * int(xr) + 1), dtype=bool)
        ma[bot-lolim:top-lolim, lft-lelim:rgt-lelim] = mask[bot:top, lft:rgt]
        ret += (ma, )

    if uncd is not None:
        un = np.zeros((2*int(yr)+1, 2*int(xr)+1)) + np.amax(uncd[bot:top,
                                                                 lft:rgt])
        un[bot - lolim:top-lolim, lft-lelim:rgt-lelim] = uncd[bot:top,
                                                              lft:rgt]
        ret += (un, )

    if len(ret) == 0:
        ret = ret[0]
    return ret


def pasteimage(data, subim, dy_, syx=(None, None)):
    """
    Inserts the subim array into data, the data coordinates (dyc,dxc)
    will match the subim coordinates (syc,sxc). The arrays can have not
    overlapping pixels.

    Parameters
    ----------
    data : 2D ndarray
        Image where subim will be inserted.
    subim : 2D ndarray
        Image so be inserted.
    dy_ : 2 elements scalar tuple
        dyc, dxc. Position in data that will match the (syc,sxc) position
        of subim.
    syx: 2 elements scalar tuple
        syc, sxc. Semi-length of the extracted image. if not specified,
        (syc,sxc) will be the center of subim.

    Returns
    -------
    The data array with the subim array inserted, according to the
    given coordinates.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> from imageedit import *
        >>> import numpy as np
        >>> # Create an array and a subimage array to past in.
        >>> data  = np.zeros((5,5), int)
        >>> subim = np.ones( (3,3), int)
        >>> subim[1,1] = 2
        >>> print(data)
        [[0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]
        [0 0 0 0 0]]
        >>> print(subim)
        [[1 1 1]
        [1 2 1]
        [1 1 1]]

        >>> # Define the matching coordinates
        >>> dyc,dxc = 3,1
        >>> syc,sxc = 1,1
        >>> # Paste subim into data
        >>> pasteimage(data, subim, (dyc,dxc), (syc,sxc))
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 2, 1, 0, 0],
        [1, 1, 1, 0, 0]]

        >>> # Paste subim into data without a complete overlap between images
        >>> data    = np.zeros((5,5), int)
        >>> dyc,dxc = 2,5
        >>> pasteimage(data, subim, (dyc,dxc), (syc,sxc))
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]
    """
    (dyc, dxc) = dy_
    (syc, sxc) = syx
    # (None,None) = Nne

    # Shape of the arrays
    dny, dnx = np.shape(data)
    sny, snx = np.shape(subim)

    if (syc, sxc) == (None, None):
        syc, sxc = sny / 2, snx / 2

    # left limits:
    led = dxc - sxc
    if led > dnx:  # the entire subimage is out of bounds
        return data

    les = np.amax([0, -led])  # left lim of subimage
    led = np.amax([0, led])  # left lim of data

    # right limits:
    rid = dxc + snx - sxc
    if rid < 0:  # the entire subimage is out of bounds
        return data

    ris = np.amin([snx, dnx - dxc + sxc])  # right lim of subimage
    rid = np.amin([dnx, rid])  # right lim of data

    # lower limits:
    lod = dyc - syc
    if lod > dny:  # the entire subimage is out of bounds
        return data

    los = np.amax([0, -lod])  # lower lim of subimage
    lod = np.amax([0, lod])  # lower lim of data

    # right limits:
    upd = dyc + sny - syc
    if upd < 0:  # the entire subimage is out of bounds
        return data

    ups = np.amin([sny, dny - dyc + syc])  # right lim of subimage
    upd = np.amin([dny, upd])  # right lim of data

    data[lod:upd, led:rid] = subim[los:ups, les:ris]

    return data
