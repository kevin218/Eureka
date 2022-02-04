import numpy as np
from astropy.convolution import Box1DKernel, convolve
from astropy.stats import sigma_clip
 
def clip_outliers(data, log, wavelength, sigma=10, box_width=5, maxiters=5, fill_value='mask', verbose=False):
    '''Find outliers in 1D time series.
  
    Be careful when using this function on a time-series with known astrophysical variations. The variable
    box_width should be set to be significantly smaller than any astrophysical variation timescales otherwise
    these signals may be clipped.
   
    Parameters
    ----------
    data: ndarray (1D, float)
        The input array in which to identify outliers
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    wavelength: float
        The wavelength currently under consideration.
    sigma: float
        The number of sigmas a point must be from the rolling mean to be considered an outlier
    box_width: int
        The width of the box-car filter (used to calculated the rolling median) in units of number of data points
    maxiters: int
        The number of iterations of sigma clipping that should be performed.
    fill_value: string or float
        Either the string 'mask' to mask the outlier values, 'boxcar' to replace data with the mean from the box-car filter, or a constant float-type fill value.
  
    Returns
    -------
    data:   ndarray (1D, boolean)
        An array with the same dimensions as the input array with outliers replaced with fill_value.
  
    Notes
    -----
    History:
  
    - Jan 29-31, 2022 Taylor Bell
        Initial version, added logging
    '''
    kernel = Box1DKernel(box_width)
    # Compute the moving mean
    smoothed_data = convolve(data, kernel, boundary='extend')
    # Compare data to the moving mean (to remove astrophysical signals)
    residuals = data-smoothed_data
    # Sigma clip residuals to find bad points in data
    residuals = sigma_clip(residuals, sigma=sigma, maxiters=maxiters)
    outliers = np.ma.getmaskarray(residuals)
  
    if np.any(outliers) and verbose:
        log.writelog('Identified {} outliers for wavelength {}'.format(np.sum(outliers), wavelength))
  
    # Replace clipped data
    if fill_value=='mask':
        data = np.ma.masked_array(data, outliers)
    elif fill_value=='boxcar':
        data = replace_moving_mean(data, outliers, kernel)
    else:
        data[outliers] = fill_value
  
    return data, np.sum(outliers)
 
def replace_moving_mean(data, outliers, kernel):
    '''Replace clipped values with the mean from a moving mean.
   
    Parameters
    ----------
    data: ndarray (1D, float)
        The input array in which to replace outliers
    outliers: ndarray (1D, bool)
        The input array in which to replace outliers
    kernel: astropy.convolution.Kernel1D
        The kernel used to compute the moving mean.
  
    Returns
    -------
    data:   ndarray (boolean)
        An array with the same dimensions as the input array with outliers replaced with fill_value.
  
    Notes
    -----
    History:
  
    - Jan 29, 2022 Taylor Bell
        Initial version
    '''
    # First set outliers to NaN so they don't bias moving mean
    data[outliers] = np.nan
    smoothed_data = convolve(data, kernel, boundary='extend')
    # Replace outliers with value of moving mean
    data[outliers] = smoothed_data[outliers]
  
    return data
 