import numpy as np
from . import gaussian as g


# Stellar Centroiding Routines
# ----------------------------
# This package contains a collection of routines related
# to the finding of the centroid of a stellar image.

# Contents:
#   ctrguess : Guesses the initial parameters of the
#              stellar centroid in a given image.
#   ctrgauss : Performs centroiding by fitting a 2D Gaussian
#              function to a given image.


def ctrguess(data, mask=None, guess=None):
    '''
    Calculates crude initial guesses of parameters required for
    Gaussian centroiding of stellar images.

    Speciffically, this function guesses the flux of the center
    of a star, the array indices of this location, and a rough guess
    of the width of the associated PSF.  This method is not robust
    to bad pixels or any other outlying values.

    Parameters
    ----------
    data : ndarray (2D)
        The image in the form of a 2D array containing the star
        to be centroided.  Works best if this is a small subarray
        of the actual data image.
    mask : ndarray (2D); optional
        A boolean mask with the same shape as the stellar image, where True
        values will be masked. Defaults to None, where only non-finite values
        are masked.
    guess : array_like; optional
        The initial guess of the position of the star.  Has the form
        (y, x) of the guess center.

    Returns
    -------
    ght : scalar
        The rough estimate of the height (or max flux) of the
        stars PSF.
    gwd : tuple
        The guessed width of the PSF, in the form (gwdy, gwdx)
        where `gwdy` and `gwdx` are the y and x widths
        respectively.
    gct : tuple
        The guessed center of the PSF, in the form (gcty, gctx)
        where `gcty` and `gctx` are the y and x center indices
        respectively.

    Notes
    -----
    Logic adapted from `gaussian.py`
    '''
    # Default mask: only non-finite values are bad
    if mask is None:
        mask = ~np.isfinite(data)

    # Apply the mask
    data = np.ma.masked_where(mask, data)

    # Center position guess, looking the max value
    if guess is None:
        gcenter = np.unravel_index(np.ma.argmax(data), np.shape(data))
    else:
        gcenter = int(guess[0]), int(guess[1])
    gheight = data[gcenter]  # height guess

    # sum of the number of pixels that are greater than two
    # sigma of the values in the x and y direction. This
    # gives a (very) rough guess, in pixels, how wide the PSF is.
    sigma = np.array([np.ma.std(data[:, gcenter[1]]),  # y std (of cent. col.)
                      np.ma.std(data[gcenter[0], :])])  # x std (of cent. row)

    gwidth = (np.ma.sum((data)[:, gcenter[1]] > 2*sigma[0]),
              np.ma.sum((data)[gcenter[0], :] > 2*sigma[1]))

    return (gwidth, gcenter, gheight)


def ctrgauss(data, guess=None, mask=None, indarr=None, trim=None):
    '''Finds and records the stellar centroid of a set of images by
    fitting a two dimensional Gaussian function to the data.

    It does not find the average centroid, but instead records
    the centroid of each image in the supplied frame parameters
    array at the supplied indices.  The frame parameters array
    is assumed to have the same number of rows as the number
    of frames in the data cube.

    Parameters
    ----------
    data : ndarray (2D)
        The stellar image.
    guess : array_like; optional
        The initial guess of the position of the star.  Has the form
        (y, x) of the guess center. If None, will call the ctrguess function.
    mask : ndarray (2D); optional
        A boolean mask with the same shape as the stellar image, where True
        values will be masked. Defaults to None, where only non-finite values
        are masked.
    indarr : array_like; optional
        The indices of the x and y center columns of the frame
        parameters and the width index.  Defaults to 4, 5, and 6
        respectively.
    trim : Scalar (positive); optional
        If trim!=None, trims the image in a box of 2*trim pixels around
        the guess center. Must be !=None for 'col' method.

    Returns
    -------
    center : y, x
        The updated frame parameters array.  Contains the centers
        of each star in each image and their average width.
    '''
    # Default mask: only non-finite values are bad
    if mask is None:
        mask = ~np.isfinite(data)

    # Apply the mask
    data = np.ma.masked_where(mask, data)

    if guess is None:
        fitguess = ctrguess(data, mask, guess)
        guess = fitguess[1]

    # the pixel of the center
    roundguess = np.round(guess)

    # Trim the image around the star if requested
    if trim is not None:
        image = data[roundguess[0]-trim:roundguess[0]+trim,
                     roundguess[1]-trim:roundguess[1]+trim]
        mask = mask[roundguess[0]-trim:roundguess[0]+trim,
                    roundguess[1]-trim:roundguess[1]+trim]
        loc = (trim, trim)
    else:
        image = np.ma.copy(data)
        loc = roundguess

    # Subtract the median to fit a gaussian.
    image -= np.ma.median(image)

    fitguess = ((1.0, 1.0), loc, image[loc[0], loc[1]])

    if indarr is None:
        indarr = np.indices(np.shape(image))

    # Fit the gaussian:
    p, err = g.fitgaussian(image, indarr, guess=fitguess, mask=mask)
    # fw = p[0:2]
    fc = p[2:4]
    # fh = p[4]

    return fc + roundguess - loc
