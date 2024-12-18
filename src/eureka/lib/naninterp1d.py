# Interpolate over NaNs in 1D

import numpy as np


def naninterp1d(y, replace_val=0):
    """
    Interpolate over NaNs in 1D array.

    Parameters
    ----------
    y : 1D array
        Data array with possible NaNs
    replace_val : float, optional
        Value to use when entire dataset is NaNs (default is 0).

    Returns
    -------
    y : 1D array
        Data array without NaNs
    """
    nans, x = nan_helper(y)
    if len(y[~nans]) > 0:
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    else:
        y[nans] = replace_val
    return y


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    y : 1D array
        Data array with possible NaNs

    Returns
    -------
    nans : 1d array
        logical indices of NaNs
    index : lambda function
        A function, with signature indices = index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices.

    Source
    ------
        https://stackoverflow.com/questions/6518811/
        interpolate-nan-values-in-a-numpy-array
    """
    return np.isnan(y), lambda z: np.nonzero(z)[0]
