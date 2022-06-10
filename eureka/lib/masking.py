import numpy as np
from scipy.interpolate import NearestNDInterpolator

__all__ = ['interpolating_row', 'data_quality_mask',
           'interpolating_image']


def interpolating_image(data, mask):
    """
    Uses `scipy.interpolate.NearestNDInterpolator` to
    fill in bad pixels/cosmic rays/whichever mask you
    decide to pass in.

    Parameters
    ----------
    data : np.ndarray
       Image frame.
    mask : np.ndarray
       Mask of integers or boolean values, where values
       greater than 0/True are bad pixels.

    Returns
    -------
    cleaned : np.ndarray
       Array of shape `data` which now has interpolated
       values over the bad masked pixels.
    """
    def interpolate(d, m):
        try:
            m = ~m
        except:
            m = m > 0

        x, y = np.meshgrid(np.arange(d.shape[1]),
                           np.arange(d.shape[0]))
        xym = np.vstack((np.ravel(x[m]), np.ravel(y[m]))).T
        data = np.ravel(d[m])
        interp = NearestNDInterpolator(xym, data)
        return interp(np.ravel(x), np.ravel(y)).reshape(d.shape)

    cleaned = np.zeros(data.shape)

    if len(data.shape) == 3:
        for i in range(len(data)):
            cleaned[i] = interpolate(data[i], mask[i])
    else:
        cleaned = interpolate(data, mask)
    return cleaned


def interpolating_row(data, mask, reg=2, arrtype='data'):
    """
    Fills in masked pixel values with either a median value from
    surrounding pixels along the row, interpolating values,
    or filling with 0.

    Parameters
    ----------
    data : np.ndarray
       Image frame.
    mask : np.ndarray
       Mask of integers, where values greater than 0 are bad
       pixels.
    reg : int, optional
       The number of pixels along the row to interpolate over.
       Default is 2.
    arrtype : str, optional
       Array type. Options are `data` and `var`, where `data`
       results in a median interpolated value to fill in and
       `var` sets the bad pixels equal to 0. Default is `data`.

    Returns
    -------
    interp : np.ndarray
       Image where the bad pixels are filled in with the
       appropriate values. Should return the same shape
       as `data`.
    """
    nanx, nany = np.where(mask > 0)
    interp = np.zeros(data.shape)

    for loc in np.array([nanx, nany]).T:
        if loc[0] < reg:
            y = np.arange(0, reg*2, 1, dtype=int)
        elif loc[0] > data.shape[0]-reg:
            y = np.arange(loc[0] - reg, data.shape[0], 1, dtype=int)
        else:
            y = np.arange(loc[0] - reg, loc[0] + reg, 1, dtype=int)

        ind = np.where(y == loc[0])[0]
        y = np.delete(y, ind)
        if arrtype == 'data':
            newval = np.nanmedian(data[y, loc[1]])
        else:
            newval = 0.0
        interp[loc[0], loc[1]] = newval + 0.0
    return interp


def data_quality_mask(dq):
    """
    Masks all pixels that are not normal (value != 0)
    or are reference pixels (value == 2147483648).

    Parameters
    ----------
    dq : np.array
       Array of data quality values.

    Returns
    -------
    dq_mask : np.array
       Boolean array masking where the bad pixels are.
    """
    dq_mask = np.ones(dq.shape, dtype=bool)

    if len(dq.shape) == 3:
        for i in range(len(dq)):
            x, y = np.where((dq[i] == 0) | (dq[i] == 2147483648))
            dq_mask[i, x, y] = False

    elif len(dq.shape) == 2:
        x, y = np.where((dq[i] == 0) | (dq[i] == 2147483648))
        dq_mask[x, y] = False
    else:
        return('Data quality array should be 2D.')

    return ~dq_mask
