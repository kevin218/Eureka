# Determine source position for data where it's not in the header (MIRI)

import numpy as np
from scipy.optimize import curve_fit
from . import plots_s3


def source_pos(data, meta, m, header=False):
    '''Make image+background plot.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    header : bool; optional
        If True, use the source position in the FITS header.
        Defaults to False.

    Returns
    -------
    src_ypos : int
        The central position of the star.
    '''
    if header:
        src_ypos = data.attrs['shdr']['SRCYPOS'] - meta.ywindow[0]
    elif meta.src_pos_type == 'weighted':
        # find the source location using a flux-weighted mean approach
        src_ypos = source_pos_FWM(data.flux.values, meta, m)
    elif meta.src_pos_type == 'gaussian':
        # find the source location using a gaussian fit
        src_ypos = source_pos_gauss(data.flux.values, meta, m)
    elif meta.src_pos_type == 'hst':
        src_ypos = data.guess.values[0]
    else:
        # brightest row for source location
        src_ypos = source_pos_max(data.flux.values, meta, m)

    return int(round(src_ypos))


def source_pos_max(flux, meta, m, plot=True):
    '''A simple function to find the brightest row for source location

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    plot : bool; optional
        If True, plot the source position determination.
        Defaults to True.

    Returns
    -------
    y_pos : int
        The central position of the star.

    Notes
    -----
    History:

    - 6/24/21 Megan Mansfield
        Initial version
    - 2021-07-14 Sebastian Zieba
        Modified
    '''

    x_dim = flux.shape[1]

    sum_row = np.sum(flux[0], axis=1)
    pos_max = np.argmax(sum_row)

    y_pixels = np.arange(0, x_dim)

    # Diagnostic plot
    if meta.isplots_S3 >= 3 and plot:
        plots_s3.source_position(meta, x_dim, pos_max, m, y_pixels=y_pixels,
                                 sum_row=sum_row)

    return pos_max


def source_pos_FWM(flux, meta, m):
    '''An alternative function to find the source location using a
    flux-weighted mean approach.

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.

    Returns
    -------
    y_pos : float
        The central position of the star.

    Notes
    -----
    History:

    - 2021-06-24 Taylor Bell
        Initial version
    - 2021-07-14 Sebastian Zieba
        Modified
    '''

    x_dim = flux.shape[1]

    pos_max = source_pos_max(flux, meta, m, plot=False)

    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

    sum_row = np.sum(flux[0],
                     axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
    sum_row -= (sum_row[0]+sum_row[-1])/2

    y_pos = np.sum(sum_row * y_pixels) / np.sum(sum_row)

    # Diagnostic plot
    if meta.isplots_S3 >= 3:
        plots_s3.source_position(meta, x_dim, pos_max, m, isFWM=True,
                                 y_pixels=y_pixels, sum_row=sum_row,
                                 y_pos=y_pos)

    return y_pos


def gauss(x, a, x0, sigma, off):
    '''A function to find the source location using a Gaussian fit.

    Parameters
    ----------
    x : ndarray
        The positions at which to evaluate the Gaussian.
    a : float
        The amplitude of the Gaussian.
    x0 : float
        The centre point of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.
    off : float
        A vertical offset in the Gaussian.

    Returns
    -------
    gaussian : ndarray
        The 1D Gaussian evaluated at the points x, in the same shape as x.

    Notes
    -----
    History:

    - 2021-07-14 Sebastian Zieba
        Initial version
    - 2021-10-15 Taylor Bell
        Separated this into its own function to allow it to be used elsewhere.
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+off


def source_pos_gauss(flux, meta, m):
    '''A function to find the source location using a gaussian fit.

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.

    Returns
    -------
    y_pos : float
        The central position of the star.

    Notes
    -----
    History:

    - 2021-07-14 Sebastian Zieba
        Initial version
    - 2021-10-15 Taylor Bell
        Tweaked to allow for cleaner plots_s3.py
    '''
    x_dim = flux.shape[1]

    # Data cutout around the maximum row
    pos_max = source_pos_max(flux, meta, m, plot=False)
    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
    sum_row = np.sum(flux[0],
                     axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

    # Initial Guesses
    sigma0 = np.sqrt(np.sum(sum_row*(y_pixels-pos_max)**2)/np.sum(sum_row))
    p0 = [np.max(sum_row), pos_max, sigma0, np.median(sum_row)]

    # Fit
    popt, pcov = curve_fit(gauss, y_pixels, sum_row, p0)

    # Diagnostic plot
    if meta.isplots_S3 >= 3:
        plots_s3.source_position(meta, x_dim, pos_max, m, isgauss=True,
                                 y_pixels=y_pixels, sum_row=sum_row,
                                 popt=popt)

    return popt[1]
