import numpy as np


def medstddev(data, mask=None, medi=False, axis=0):
    """Compute the stddev with respect to the median.

    This is rather than the standard method of using the mean.

    Parameters
    ----------
    data : ndarray
        An array from which to caculate the median standard deviation.
    mask : 1D ndarray; optional
        Boolean mask indicating the bad values with True.
        Same shape as data. Defaults to None.
    medi : boolean; optional
        If True return a tuple with (stddev, median) of data. Defaults
        to False.
    axis : int; optional
        The axis along wich the median std deviation is calculated.
        Defaults to 0.

    Returns
    -------
    float
        The stadard deviation.
    float; optional
        The median; only returned if medi==True.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> import medstdev as m
        >>> a  = np.array([1,3,4,5,6,7,7])
        >>> std, med = m.medstddev(a, medi=True)
        >>> print(median(a))
        5.0
        >>> print(med)
        5.0
        >>> print(std)
        2.2360679775

        >>> # use masks
        >>> a    = np.array([1,3,4,5,6,7,7])
        >>> mask = np.array([False,False,False,True,True,True,True])
        >>> std, med = m.medstddev(a, mask, medi=True)
        >>> print(std)
        1.58113883008
        >>> print(med)
        3.0

        >>> # automatically mask invalid values
        >>> a = np.array([np.nan, 1, 4, np.inf, 6])
        >>> std, med = m.medstddev(a, medi=True)
        >>> print(std, med)
        (2.5495097567963922, 4.0)

        >>> # critical cases:
        >>> # only one value, return std = 0.0
        >>> a    = np.array([1, 4, 6])
        >>> mask = np.array([True, True, False])
        >>> std, med = m.medstddev(a, mask, medi=True)
        >>> print(std, med)
        (0.0, 6.0)

        >>> # no good values, return std = nan, med = nan
        >>> mask[-1] = True
        >>> std, med = m.medstddev(a, mask, medi=True)
        >>> print(std, med)
        (nan, nan)

    Notes
    -----
    MEANSTDDEV calculates the median, subtracts it from each value of
    X, then uses this residual to calculate the standard deviation.

    The numerically-stable method for calculating the variance from
    moment.pro doesn't work for the median standard deviation.  It
    only works for the mean, because by definition the residuals from
    the mean add to zero.

    History:

    - 2005-01-18  statia
        Written by Statia Luszcz.
    - 2005-01-19  statia
        Updated variance calculation according to algorithm in moment.pro,
        added medi keyword.
    - 2005-01-20  Joe Harrington, Cornell, jh@oobleck.astro.cornell.edu
        Header update.  Removed algorithm from moment.pro because it
        doesn't work for the median.  Added /double.
    - 2010-11-05  patricio  pcubillos@fulbrightmail.org
        Converted to python, documented.
    - 2022-04-11  Taylor James Bell
        Efficiently using numpy axes
    """
    # Default mask: only non-finite values are bad
    if mask is None:
        mask = ~np.isfinite(data)

    # Apply the mask
    data = np.ma.masked_where(mask, data)

    # number of good values:
    ngood = np.sum(~mask, axis=axis)

    # calculate median of good values:
    median = np.ma.median(data, axis=axis)
    # residuals is data - median, masked values don't count:
    residuals = data - median
    # calculate standar deviation:
    with np.errstate(divide='ignore', invalid='ignore'):
        std = np.ma.std(residuals, axis=axis, ddof=1)

    # Convert masked arrays to just arrays
    std = np.array(std)
    median = np.array(median)
    if std.shape == ():
        # If just a single value, make sure using a shaped array
        std = std.reshape(-1)
        median = median.reshape(-1)

    # critical case fixes:
    if np.any(ngood == 0):
        std[np.where(ngood == 0)] = np.nan
        median[np.where(ngood == 0)] = np.nan
    if np.any(ngood == 1):
        std[np.where(ngood == 1)] = 0.

    if len(std) == 1:
        std = std[0]
        median = median[0]

    # return statement:
    if medi:
        return (std, median)
    return std
