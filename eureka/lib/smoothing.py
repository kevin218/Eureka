import numpy as np


def gauss_kernel_mask2(ny_nx, sy_sx, j_i, mask):
    """Create a 2D Gaussian kernel mask.

    Parameters
    ----------
    ny_nx : list/ndarray
        The ny and nx values.
    sy_sx : list/ndarray
        The sy and sx values.
    j_i : list/ndarray
        The j and i values.
    mask : ndarray
        The current mask.

    Returns
    -------
    ndarray
        The Gaussian kernel.
    """
    j, i = j_i
    sy, sx = sy_sx
    ny = int(ny_nx[0])
    nx = int(ny_nx[1])
    y, x = 1.*np.mgrid[-ny:ny+1, -nx:nx+1]
    # Create new mask with padding
    sizey, sizex = mask.shape
    newmask = np.zeros(np.array((sizey, sizex)) + 2*np.array((ny, nx)))
    # Copy the mask into the middle of the new mask, leaving padding as zeros
    newmask[ny:-ny, nx:-nx] = mask
    gauss = (np.exp(-(0.5*(x/sx)**2 + 0.5*(y/sy)**2))
             * newmask[j:j+2*ny+1, i:i+2*nx+1])
    gsum = gauss.sum()
    if gsum > 0:
        kernel = gauss/gsum
    else:
        kernel = 0.

    return kernel
