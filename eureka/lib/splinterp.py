import scipy.interpolate as si


def splinterp(x2, x, y):
    """This function implements the methods splrep and splev of the
    module scipy.interpolate

    Parameters
    ----------
    x2 : 1D array_like
        array of points at which to return the value of the
        smoothed spline or its derivatives
    x : array_like
        The x data points defining a curve y = f(x).
    y : array_like
        The y data points defining a curve y = f(x).

    Returns
    -------
    array_like
        An array of values representing the spline function or curve.
        If tck was returned from splrep, then this is a list of arrays
        representing the curve in N-dimensional space.


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.arange(21)/20.0 * 2.0 * np.pi
    >>> y = np.sin(x)
    >>> x2 = np.arange(41)/40.0 *2.0 * np.pi

    >>> y2 = splinterp(x2, x, y)
    >>> plt.plot(x2,y2)
    """
    tck = si.splrep(x, y)
    y2 = si.splev(x2, tck)
    return y2
