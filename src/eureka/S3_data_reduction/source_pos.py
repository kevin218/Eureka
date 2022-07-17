# Determine source position for data where it's not in the header (MIRI)

import numpy as np
from scipy.optimize import curve_fit
from . import plots_s3


def source_pos(data, meta, m, integ=0):
    '''Make image+background plot.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    integ : int, optional
        The integration number.
        Default is 0 (first integration)


    Returns
    -------
    src_ypos : int
        The central position of the star.
    src_ypos_exact : float
        The exact (not rounded) central position of the star.
    src_ypos_width : float
        If gaussian fit, the std of the Gaussian fitted to the image
        Otherwise, array of zeros.
        
    Notes
    -----
    History:
    
    - 2022-07-11 Caroline Piaulet
        Enable recording of the width if the source is fitted with a Gaussian
        + add an option to fit any integration (not hardcoded to be the first)
    '''
    # Mask any clipped values
    flux = np.ma.masked_where(~data.mask.values, data.flux.values)

    if meta.src_pos_type == 'header':
        if 'SRCYPOS' not in data.attrs['shdr']:
            raise AttributeError('There is no SRCYPOS in the FITS header. '
                                 'You must select a different value for '
                                 'meta.src_pos_type')
        src_ypos = data.attrs['shdr']['SRCYPOS'] - meta.ywindow[0]
    elif meta.src_pos_type == 'weighted':
        # find the source location using a flux-weighted mean approach
        src_ypos = source_pos_FWM(flux, meta, m, integ=integ)
    elif meta.src_pos_type == 'gaussian':
        # find the source location using a gaussian fit
        src_ypos, src_ywidth = source_pos_gauss(flux, meta, m,
                                                integ=integ)
    elif meta.src_pos_type == 'hst':
        src_ypos = data.guess.values[0]
    else:
        # brightest row for source location
        src_ypos = source_pos_max(flux, meta, m, integ=integ)

    if meta.src_pos_type == 'gaussian':
        return int(round(src_ypos)), src_ypos, src_ywidth
    else:
        return int(round(src_ypos)), src_ypos, np.zeros_like(src_ypos)


def source_pos_max(flux, meta, m, integ=0, plot=True):
    '''A simple function to find the brightest row for source location

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    integ : int, optional
        The integration number.
        Default is 0 (first integration)
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
    - July 11, 2022 Caroline Piaulet
        Add option to fit any integration (not hardcoded to be the first)
    '''

    x_dim = flux.shape[1]

    sum_row = np.ma.sum(flux[integ], axis=1)
    pos_max = np.ma.argmax(sum_row)

    # Diagnostic plot
    if meta.isplots_S3 >= 3 and plot:
        y_pixels = np.arange(0, x_dim)
        plots_s3.source_position(meta, x_dim, pos_max, m, y_pixels=y_pixels,
                                 sum_row=sum_row)

    return pos_max


def source_pos_FWM(flux, meta, m, integ=0):
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
    integ : int, optional
        The integration number.
        Default is 0 (first integration)

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
    - 2022-07-11 Caroline Piaulet
        Add option to fit any integration (not hardcoded to be the first)
    '''

    x_dim = flux.shape[1]

    pos_max = source_pos_max(flux, meta, m, plot=False)

    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

    sum_row = np.ma.sum(flux[integ],
                        axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
    sum_row -= (sum_row[0]+sum_row[-1])/2

    y_pos = np.ma.sum(sum_row * y_pixels)/np.ma.sum(sum_row)

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


def source_pos_gauss(flux, meta, m, integ=0):
    '''A function to find the source location using a gaussian fit.

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    integ : int, optional
        The integration number.
        Default is 0 (first integration)

    Returns
    -------
    y_pos : float
        The central position of the star.
    y_width : int
        The std of the fitted Gaussian.

    Notes
    -----
    History:

    - 2021-07-14 Sebastian Zieba
        Initial version
    - 2021-10-15 Taylor Bell
        Tweaked to allow for cleaner plots_s3.py
    - 2022-07-11 Caroline Piaulet
        Enable recording of the width if the source is fitted with a Gaussian
        + add an option to fit any integration (not hardcoded to be the first)
    '''
    x_dim = flux.shape[1]

    # Data cutout around the maximum row
    pos_max = source_pos_max(flux, meta, m, integ=integ, plot=False)
    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
    sum_row = np.ma.sum(flux[integ],
                        axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

    # Initial Guesses
    sigma0 = np.ma.sqrt(np.ma.sum(sum_row*(y_pixels-pos_max)**2) /
                        np.ma.sum(sum_row))

    p0 = [np.ma.max(sum_row), pos_max, sigma0, np.ma.median(sum_row)]

    # Fit
    popt, pcov = curve_fit(gauss, y_pixels, sum_row, p0)

    # Diagnostic plot
    if meta.isplots_S3 >= 3:
        plots_s3.source_position(meta, x_dim, pos_max, m, isgauss=True,
                                 y_pixels=y_pixels, sum_row=sum_row,
                                 popt=popt)
    return popt[1], popt[2]
