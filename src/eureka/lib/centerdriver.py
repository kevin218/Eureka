import numpy as np
from . import imageedit as ie
from . import gaussian as g
from . import gaussian_min as gmin
from ..S3_data_reduction import plots_s3


def centerdriver(method, data, guess, trim, radius, size, i, m, meta,
                 saved_ref_median_frame,
                 mask=None, uncd=None, fitbg=1, maskstar=True,
                 expand=5.0, psf=None, psfctr=None):
    """
    Use the center method to find the center of a star in data, starting
    from position guess.

    Parameters
    ----------
    method : string
        Name of the centering method to use.
    data : 2D ndarray
        Array containing the star image.
    guess : 2 elements 1D array
        y, x initial guess position of the target.
    trim : integer
        Semi-lenght of the box around the target that will be trimmed.
    radius : float
        least asymmetry parameter. See err_fasym_c.
    size : float
        least asymmetry parameter. See err_fasym_c.
    mask : 2D ndarray
        A mask array of bad pixels. Same shape of data.
    uncd : 2D ndarray
        An array containing the uncertainty values of data.
        Same shape of data.
    saved_ref_median_frame : ndarray
        Stored median frame of the first batch.

    Returns
    -------
    tuple
        A (y,x) tuple (scalars) with the coordinates of center of the target
        in data.
    ndarray
        The median of the first batch to be used as the refrence
        frame for the first centroid location guess in the mgmc method.

    Notes
    -----
    History:

    23-11-2010 patricio   Written by Patricio Cubillos
                          pcubillos@fulbrightmail.org
    2-24-2023  Isaac      Edited by Isaac edelman
                          Added new centroiding method
                          called mgmc_pri and mgmc_sec
    """

    extra = []

    # Default mask: all good
    if mask is None:
        mask = np.ones(np.shape(data))

    # Default uncertainties: flat image
    if uncd is None:
        uncd = np.ones(np.shape(data))

    if method in ['fgc', 'fgc_sec']:
        # Trim the image if requested
        if trim != 0:
            # Integer part of center
            cen = np.rint(guess)
            # Center in the trimed image
            loc = (trim, trim)
            # Do the trim:
            img, msk, err = ie.trimimage(data, cen, loc, mask=mask, uncd=uncd)
        else:
            cen = np.array([0, 0])
            loc = np.rint(guess)
            img, msk, err = data, mask, uncd
        weights = 1.0 / np.abs(err)
    else:
        trim = 0
        img, msk, err = data, mask, uncd
        loc = guess
        cen = np.array([0, 0])
        # Subtract median BG because photutils sometimes has a hard time
        # fitting for a constant offset
        img -= np.nanmedian(img)

    # If all data is bad:
    if not np.any(msk):
        raise Exception('Bad Frame Exception!')

    # Get the center with one of the methods:
    refrence_median_frame = None
    if method in ['fgc', 'fgc_sec']:
        sy, sx, y, x = g.fitgaussian(img, yxguess=loc, mask=msk,
                                     weights=weights,
                                     fitbg=fitbg, maskg=maskstar)[0][0:4]
        extra = sy, sx  # Gaussian 1-sigma half-widths
    elif method == 'mgmc_pri':
        # Median frame creation + first centroid
        x, y, refrence_median_frame = gmin.pri_cent(img, meta,
                                                    saved_ref_median_frame)
    elif method == 'mgmc_sec':
        # Second enhanced centroid position + gaussian widths
        sy, sx, y, x = gmin.mingauss(img, yxguess=loc, meta=meta)
        extra = sy, sx  # Gaussian 1-sigma half-widths

    # only plot when we do the second fit
    if (meta.isplots_S3 >= 5 and method[-4:] == '_sec' and i < meta.nplots):
        plots_s3.phot_centroid_fgc(img, x, y, sx, sy, i, m, meta)

    # Make trimming correction and return
    return ((y, x) + cen - trim), extra, refrence_median_frame
