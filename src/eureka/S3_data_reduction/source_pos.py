# Determine source position for data where it's not in the header (MIRI)

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import multiprocessing as mp
from . import plots_s3


def source_pos_wrapper(data, meta, log, m, integ=0):
    '''Determine the source position for many frames.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.
    integ : int or None; optional
        The integration number. Default is 0 (first integration).
        If set to None, the source position and width for each frame will be
        calculated and stored in data.centroid_y and data.centroid_sy.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.

    Notes
    -----
    History:

    - 2022-07-18, Taylor J Bell
        Added source_pos_wrapper to allow multiple frames to get
        source positions in parallel.
    '''
    # Mask any clipped values
    flux = np.ma.masked_where(~data.mask.values, data.flux.values)

    if meta.src_pos_type == 'hst':
        guess = data.guess.values[0]
    else:
        guess = None

    if integ is None:
        log.writelog("  Recording y position and width for all "
                     "integrations...", mute=(not meta.verbose))
        # Get the source position of every frame
        src_ypos_exact = np.zeros_like(data["time"])
        src_ypos_width = np.zeros_like(data["time"])

        # Write source_positions
        def writePos(arg):
            src_ypos, src_yexact, src_ywidth, n = arg
            src_ypos_exact[n] = src_yexact
            src_ypos_width[n] = src_ywidth
            return

        if meta.ncpu == 1:
            # Only 1 CPU
            iterfn = range(meta.int_start, meta.n_int)
            if meta.verbose:
                iterfn = tqdm(iterfn)
            for n in iterfn:
                writePos(source_pos(flux[n], meta, data.attrs['shdr'],
                                    m, n, False, guess))
        else:
            # Multiple CPUs
            pool = mp.Pool(meta.ncpu)
            jobs = [pool.apply_async(func=source_pos,
                                     args=(flux[n], meta,
                                           data.attrs['shdr'], m,
                                           n, False, guess),
                                     callback=writePos)
                    for n in range(meta.int_start, meta.n_int)]
            pool.close()
            iterfn = jobs
            if meta.verbose:
                iterfn = tqdm(iterfn)
            for job in iterfn:
                job.get()

        data['centroid_y'] = (['time'], src_ypos_exact)
        data.centroid_y.attrs['units'] = 'pixels'
        data['centroid_sy'] = (['time'], src_ypos_width)
        data.centroid_sy.attrs['units'] = 'pixels'

        return data, meta, log
    else:
        # Get the source position of frame `integ`
        log.writelog('  Locating source position...', mute=(not meta.verbose))

        meta.src_ypos = source_pos(flux[integ], meta, data.attrs['shdr'],
                                   m, integ, True, guess)[0]

        log.writelog('    Source position on detector is row '
                     f'{meta.src_ypos}.', mute=(not meta.verbose))

        return data, meta, log


def source_pos(flux, meta, shdr, m, n, plot=True, guess=None):
    '''Determine the source position for one frames.

    Parameters
    ----------
    flux : np.ma.masked_array (2D)
        The 2D image from which to get the source position.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    shdr : astropy.io.fits.header.Header
        The science header of the file being processed.
    m : int
        The file number.
    n : int
        The integration number.
    plot : bool; optional
        If True, plot the source position determination.
        Defaults to True.
    guess : float; optional
        The guess at the source position for WFC3 data.

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
    - 2022-07-18, Taylor J Bell
        Tweaked to allow parallelized code if fitting multiple frames.
    '''
    if meta.src_pos_type == 'header':
        if 'SRCYPOS' not in shdr:
            raise AttributeError('There is no SRCYPOS in the FITS header. '
                                 'You must select a different value for '
                                 'meta.src_pos_type')
        src_ypos = shdr['SRCYPOS'] - meta.ywindow[0]
    elif meta.src_pos_type == 'weighted':
        # find the source location using a flux-weighted mean approach
        src_ypos = source_pos_FWM(flux, meta, m, n, plot)
    elif meta.src_pos_type == 'gaussian':
        # find the source location using a gaussian fit
        src_ypos, src_ywidth = source_pos_gauss(flux, meta, m, n, plot)
    elif meta.src_pos_type == 'hst':
        src_ypos = guess
    elif meta.src_pos_type == 'max':
        # brightest row for source location
        src_ypos = source_pos_median(flux, meta, m, n, plot)
    elif not isinstance(meta.src_pos_type, str):
        # manually specify source location
        src_ypos = float(meta.src_pos_type)
    else:
        # Some unrecognized string
        raise Exception(f'{meta.src_pos_type} is not a recognized source ' +
                        'position type. Options: header, gaussian, weighted,' +
                        ' max, hst, or a numeric value.')

    if meta.src_pos_type == 'gaussian':
        return int(round(src_ypos)), src_ypos, src_ywidth, n
    else:
        return int(round(src_ypos)), src_ypos, np.zeros_like(src_ypos), n


def source_pos_median(flux, meta, m, n=0, plot=True):
    '''A simple function to find the brightest row for source location

    Parameters
    ----------
    flux : ndarray
        The 2D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    n : int; optional
        The integration number. Default is 0 (first integration).
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

    - 4/27/23 Kevin Stevenson
        Modified from source_pos_max
    '''
    x_dim = flux.shape[0]

    med_row = np.ma.median(flux, axis=1)
    pos_max = np.ma.argmax(med_row)

    # Diagnostic plot
    if meta.isplots_S3 >= 1 and plot:
        y_pixels = np.arange(0, x_dim)
        plots_s3.source_position(meta, x_dim, pos_max, m, n, y_pixels=y_pixels,
                                 sum_row=med_row)

    return pos_max


def source_pos_max(flux, meta, m, n=0, plot=True):
    '''A simple function to find the brightest row for source location

    Parameters
    ----------
    flux : ndarray
        The 2D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    n : int; optional
        The integration number. Default is 0 (first integration).
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
    x_dim = flux.shape[0]

    sum_row = np.ma.sum(flux, axis=1)
    pos_max = np.ma.argmax(sum_row)

    # Diagnostic plot
    if meta.isplots_S3 >= 1 and plot:
        y_pixels = np.arange(0, x_dim)
        plots_s3.source_position(meta, x_dim, pos_max, m, n, y_pixels=y_pixels,
                                 sum_row=sum_row)

    return pos_max


def source_pos_FWM(flux, meta, m, n=0, plot=True):
    '''An alternative function to find the source location using a
    flux-weighted mean approach.

    Parameters
    ----------
    flux : ndarray
        The 2D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    n : int; optional
        The integration number. Default is 0 (first integration).
    plot : bool; optional
        If True, plot the source position determination.
        Defaults to True.

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
    x_dim = flux.shape[0]

    pos_max = source_pos_median(flux, meta, m, n=n, plot=False)

    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

    sum_row = np.ma.sum(flux, axis=1)[pos_max-meta.spec_hw:
                                      pos_max+meta.spec_hw]
    sum_row -= (sum_row[0]+sum_row[-1])/2

    y_pos = np.ma.sum(sum_row*y_pixels)/np.ma.sum(sum_row)

    # Diagnostic plot
    if meta.isplots_S3 >= 1 and plot:
        plots_s3.source_position(meta, x_dim, pos_max, m, n, isFWM=True,
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


def source_pos_gauss(flux, meta, m, n=0, plot=True):
    '''A function to find the source location using a gaussian fit.

    Parameters
    ----------
    flux : ndarray
        The 3D array of flux values.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    n : int; optional
        The integration number.
        Default is 0 (first integration)
    plot : bool; optional
        If True, plot the source position determination.
        Defaults to True.

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
    x_dim = flux.shape[0]

    # Data cutout around the maximum row
    pos_max = source_pos_median(flux, meta, m, n=n, plot=False)
    y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
    med_row = np.ma.median(flux, axis=1)[pos_max-meta.spec_hw:
                                         pos_max+meta.spec_hw]

    # Initial Guesses
    sigma0 = np.ma.sqrt(np.ma.sum(med_row*(y_pixels-pos_max)**2) /
                        np.ma.sum(med_row))

    p0 = [np.ma.max(med_row), pos_max, sigma0, np.ma.median(med_row)]

    # Fit
    popt, pcov = curve_fit(gauss, y_pixels, med_row, p0, maxfev=10000)

    # Diagnostic plot
    if meta.isplots_S3 >= 1 and plot:
        plots_s3.source_position(meta, x_dim, pos_max, m, n, isgauss=True,
                                 y_pixels=y_pixels, sum_row=med_row,
                                 popt=popt)

    return popt[1], popt[2]
