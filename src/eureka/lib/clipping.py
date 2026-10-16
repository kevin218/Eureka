import warnings

import numpy as np
from astropy.convolution import Box1DKernel, convolve
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.stats import sigma_clip
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.special import erf

__all__ = ['clip_outliers', 'rolling_median', 'gauss_removal']


def rolling_median(data, box_width, boundary='fill', fill_value=None):
    '''Compute a rolling median that ignores masked and non-finite values.

    Parameters
    ----------
    data : ndarray or np.ma.MaskedArray (1D)
        The input time series. Masked and non-finite values are ignored.
    box_width : int
        The width of the rolling window in units of number of data points.
        Even widths are rounded up to the next odd number so that the window
        is centred on each point.
    boundary : str or None; optional
        How to handle the edges of the time series. 'fill' pads the edges with
        fill_value, 'wrap' uses a periodic boundary, 'extend' repeats the
        first/last data points, and None uses only the available data points
        near the edges. Defaults to 'fill'.
    fill_value : float; optional
        The value used to pad the edges when boundary=='fill'. Defaults to
        None, in which case the median of the valid data is used.

    Returns
    -------
    ndarray (1D)
        The rolling median, which is NaN wherever a window contains no valid
        data points.
    '''
    values = np.ma.filled(np.ma.masked_invalid(data).astype(float), np.nan)
    half = int(box_width)//2

    if boundary == 'fill':
        if fill_value is None:
            fill_value = np.nanmedian(values)
        padded = np.pad(values, half, mode='constant',
                        constant_values=fill_value)
    elif boundary == 'wrap':
        padded = np.pad(values, half, mode='wrap')
    elif boundary == 'extend':
        padded = np.pad(values, half, mode='edge')
    elif boundary is None:
        padded = np.pad(values, half, mode='constant',
                        constant_values=np.nan)
    else:
        raise ValueError(f'Unknown boundary "{boundary}". Options are '
                         '"fill", "wrap", "extend", or None.')

    windows = sliding_window_view(padded, 2*half+1)
    # Fast path: scipy's median filter is exact for windows without NaNs.
    # Only windows that contain masked/NaN values need np.nanmedian
    isnan = np.isnan(padded)
    smoothed = median_filter(np.where(isnan, 0., padded), size=2*half+1,
                             mode='nearest')[half:padded.size-half]
    nan_cumsum = np.concatenate([[0], np.cumsum(isnan)])
    has_nan = (nan_cumsum[2*half+1:] - nan_cumsum[:-(2*half+1)]) > 0
    if np.any(has_nan):
        with warnings.catch_warnings():
            # All-NaN windows (e.g. long masked regions) intentionally give
            # NaN
            warnings.simplefilter('ignore', RuntimeWarning)
            smoothed[has_nan] = np.nanmedian(windows[has_nan], axis=1)

    return smoothed


def clip_outliers(data, log, wavelength, wavelength_units='microns', mask=None,
                  sigma=10, box_width=5, maxiters=5, boundary='extend',
                  fill_value='mask', verbose=False):
    '''Find outliers in 1D time series.

    Be careful when using this function on a time-series with known
    astrophysical variations. The variable box_width should be set to be
    significantly smaller than any astrophysical variation timescales otherwise
    these signals may be clipped.

    Parameters
    ----------
    data : ndarray (1D)
        The input array in which to identify outliers
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    wavelength : float
        The wavelength currently under consideration.
    wavelength_units : float
        The wavelength units currently under consideration.
    mask : ndarray (1D); optional
        A boolean mask array to use if data is not a masked array, where True
        values will be masked. Defaults to None in which case only the invalid
        values of data will be masked.
    sigma : float; optional
        The number of sigmas a point must be from the rolling median to be
        considered an outlier. Defaults to 10.
    box_width : int; optional
        The width of the window used to calculate the rolling median in
        units of number of data points (even widths are rounded up to the
        next odd number). Defaults to 5.
    maxiters : int; optional
        The number of iterations of sigma clipping that should be performed.
        Defaults to 5.
    boundary : str or None; optional
        How to handle the edges when computing the rolling median: 'fill'
        pads with the median of all data points, 'wrap' uses a periodic
        boundary, 'extend' repeats the first/last data points, and None uses
        only the available data points. Defaults to 'extend'.
    fill_value : str or float; optional
        Either the string 'mask' to mask the outlier values, 'boxcar' to
        replace data with the mean from the box-car filter, or a constant
        float-type fill value. Defaults to 'mask'.
    verbose : bool; optional
        If True, log details about the outliers found at this wavelength.
        Defaults to False.

    Returns
    -------
    data : ndarray (1D)
        An array with the same dimensions as the input array with outliers
        replaced with fill_value.
    outliers : ndarray (1D)
        A boolean array where True for values that were clipped.
    noutliers : int
        The number of outliers identified.
    '''
    data = np.ma.masked_invalid(np.ma.copy(data))
    data = np.ma.masked_where(mask, data)

    kernel = Box1DKernel(box_width)

    outliers = np.zeros(data.shape, dtype=bool)
    new_clipped = True
    i = 0
    while i < maxiters and new_clipped:
        i += 1

        # Compute the rolling median using only the points not currently
        # flagged as outliers. Unlike a moving mean, a rolling median is not
        # dragged by a strong outlier, so its good neighbours are not clipped
        clean_data = np.ma.masked_where(outliers, data)
        bound_val = np.ma.median(clean_data)  # Only used if boundary=='fill'
        smoothed_data = rolling_median(clean_data, box_width,
                                       boundary=boundary,
                                       fill_value=bound_val)
        # Compare all data to the rolling median (to remove astrophysical
        # signals). Re-evaluating previously flagged points allows any points
        # flagged in an earlier iteration to be restored if they are no
        # longer outliers
        residuals = data-smoothed_data
        # Sigma clip residuals to find bad points in data. As before, the
        # returned outliers also include any input-masked points
        residuals = sigma_clip(residuals, sigma=sigma, maxiters=maxiters,
                               cenfunc=np.ma.median)
        new_outliers = np.ma.getmaskarray(residuals)
        if np.all(new_outliers == outliers):
            new_clipped = False
        else:
            outliers = new_outliers

    if i == maxiters:
        log.writelog('WARNING: maxiters has been reached during clip_outliers '
                     'without converging!')

    if np.any(outliers):
        log.writelog(f'  Identified {np.sum(outliers)} outliers for wavelength'
                     f' {wavelength:.4f} '
                     f'{wavelength_units}', mute=(not verbose))

    # Replace clipped data
    if fill_value == 'mask':
        data = np.ma.masked_where(outliers, data)
    elif fill_value == 'boxcar':
        data = replace_moving_mean(data, outliers, kernel)
        outliers[:] = False
    else:
        data[outliers] = fill_value

    return data, outliers, np.sum(outliers)


def replace_moving_mean(data, outliers, kernel):
    '''Replace clipped values with the mean from a moving mean.

    Parameters
    ----------
    data : ndarray (1D, float)
        The input array in which to replace outliers
    outliers : ndarray (1D, bool)
        The input array in which to replace outliers
    kernel : astropy.convolution.Kernel1D
        The kernel used to compute the moving mean.

    Returns
    -------
    data : ndarray (boolean)
        An array with the same dimensions as the input array with outliers
        replaced with fill_value.
    '''
    # First set outliers to NaN so they don't bias moving mean
    data[outliers] = np.nan
    smoothed_data = convolve(data, kernel, boundary='extend')
    # Replace outliers with value of moving mean
    data[outliers] = smoothed_data[outliers]

    return data


def skewed_gaussian(x, eta=0, omega=1, alpha=0, scale=1):
    """A skewed Gaussian model.

    Parameters
    ----------
    x : ndarray
        The values at which to evaluate the skewed Gaussian.
    eta : float; optional
        The Gaussian mean. Defaults to 0.
    omega : float; optional
        The skewed normal scale. Defaults to 1.
    alpha : float; optional
        The skewed normal shape. Defaults to 0.
    scale : float; optional
        A multiplier for the skewed normal. Defaults to 1.

    Returns
    -------
    ndarray
        The skewed Gaussian model evaluated at positions x.
    """
    t = alpha*(x-eta)/omega
    Psi = 0.5*(1+erf(t/np.sqrt(2)))
    psi = 2/(omega*np.sqrt(2*np.pi))*np.exp(-(x-eta)**2/(2*omega**2))
    return (psi * Psi)*scale


def gauss_removal(img, mask, linspace, where='bkg'):
    """An additional step to remove cosmic rays.

    This fits a Gaussian to the background (or a skewed Gaussian to the
    orders) and masks data points which are above a certain sigma.

    Parameters
    ----------
    img : np.ndarray
       Single exposure image.
    mask : np.ndarray
       An approximate boolean mask for the orders, where True values are
       masked.
    linspace : array
       Sets the lower and upper bin bounds for the
       pixel values. Should be of length = 2.
    where : str; optional
       Sets where the mask is covering. Default is `bkg`.
       Other option is `order`.

    Returns
    -------
    img : np.ndarray
       The same input image, now masked for newly identified
       outliers.
    """
    weights = (~mask).astype(np.float64)
    n, bins = np.histogram(img.flatten(),
                           bins=np.linspace(linspace[0], linspace[1], 100),
                           weights=weights.flatten())
    bincenters = (bins[1:]+bins[:-1])/2

    if where == 'bkg':
        g = Gaussian1D(mean=0, amplitude=100, stddev=10)
        rmv = np.where(np.abs(bincenters) <= 5)[0]
    elif where == 'order':
        GaussianSkewed = custom_model(skewed_gaussian)
        g = GaussianSkewed(eta=0, omega=20, alpha=4, scale=100)
        rmv = np.where(np.abs(bincenters) == 0)[0]

    # finds bin centers and removes bincenter = 0 (because this bin
    #   seems to be enormous and we don't want to skew the best-fit
    bincenters, n = np.delete(bincenters, rmv), np.delete(n, rmv)

    # fit the model to the histogram bins
    fitter = LevMarLSQFitter()
    gfit = fitter(g, bincenters, n)

    masked_img = np.ma.masked_where(mask, img)

    if where == 'bkg':
        xcr, ycr = np.ma.where(np.ma.abs(masked_img) >=
                               gfit.mean+2*gfit.stddev)
    elif where == 'order':
        xcr, ycr = np.ma.where(masked_img <= gfit.eta-1*gfit.omega)
    else:
        raise ValueError(f'Unrecognized value "{where}" for argument "where".')

    # returns an image that is nan-masked
    img[xcr, ycr] = np.nan
    return img
