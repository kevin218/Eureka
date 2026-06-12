import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from ..lib import plots
from . import plots_s3

__all__ = ['BGsubtraction', 'fitbg', 'fitbg2']


def BGsubtraction(data, meta, log, m, isplots=0, group=None):
    """Does background subtraction using inst.fit_bg & background.fitbg

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of MJy/sr or electrons.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    m : int
        The current file/batch number.
    isplots : bool; optional
       Plots intermediate steps for the background fitting routine.
       Default is False.
    group : int; optional
        The group number (only applies to Stage 1).  Default is None.

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing background subtracted data.
    """
    if meta.bg_deg is None or meta.skip_bg:
        # Need to skip doing background subtraction
        log.writelog('  Skipping background subtraction...',
                     mute=(not meta.verbose))
        data['bg'] = (list(data.coords.keys()), np.zeros_like(data.flux))
        data['bg'].attrs['flux_units'] = data['flux'].attrs['flux_units']
        return data

    # Load instrument module
    if meta.inst == 'miri':
        from . import miri as inst
    elif meta.inst == 'nircam':
        from . import nircam as inst
    elif meta.inst == 'nirspec':
        from . import nirspec as inst
    elif meta.inst == 'niriss':
        from . import niriss as inst
    elif meta.inst == 'wfc3':
        from . import wfc3 as inst
    else:
        raise ValueError('Unknown instrument {}'.format(meta.inst))

    # Write background
    def writeBG(arg):
        bg_data, bg_mask, n = arg
        data['bg'][n] = bg_data
        data['mask'][n] = bg_mask
        return

    def writeBG_WFC3(arg):
        bg_data, bg_mask, datav0, datavariance, n = arg
        data['bg'][n] = bg_data
        data['mask'][n] = bg_mask
        data['v0'][n] = datav0
        data['variance'][n] = datavariance
        return

    # Compute background for each integration
    log.writelog('  Performing ' + meta.bg_dir + ' background subtraction...',
                 mute=(not meta.verbose))
    data['bg'] = (list(data.coords.keys()), np.zeros_like(data.flux))
    data['bg'].attrs['flux_units'] = data['flux'].attrs['flux_units']
    if meta.ncpu == 1:
        # Only 1 CPU
        iterfn = range(meta.int_start, meta.n_int)
        if meta.verbose:
            iterfn = tqdm(iterfn)
        for n in iterfn:
            # Fit sky background with out-of-spectra data
            if meta.inst == 'wfc3':
                writeBG_WFC3(inst.fit_bg(data.flux[n].values,
                                         data.mask[n].values,
                                         data.v0[n].values,
                                         data.variance[n].values,
                                         data.guess[n].values,
                                         n, meta, isplots))
            else:
                writeBG(inst.fit_bg(data.flux[n].values, data.mask[n].values,
                                    n, meta, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        if meta.inst == 'wfc3':
            # The WFC3 background subtraction needs a few more inputs
            # and outputs
            jobs = [pool.apply_async(func=inst.fit_bg,
                                     args=(data.flux[n].values,
                                           data.mask[n].values,
                                           data.v0[n].values,
                                           data.variance[n].values,
                                           data.guess[n].values,
                                           n, meta, isplots,),
                                     callback=writeBG_WFC3)
                    for n in range(meta.int_start, meta.n_int)]
        else:
            jobs = [pool.apply_async(func=inst.fit_bg,
                                     args=(data.flux[n].values,
                                           data.mask[n].values,
                                           n, meta, isplots,),
                                     callback=writeBG)
                    for n in range(meta.int_start, meta.n_int)]
        pool.close()
        iterfn = jobs
        if meta.verbose:
            iterfn = tqdm(iterfn)
        for job in iterfn:
            job.get()

    # 9.  Background subtraction
    # Perform background subtraction
    data['flux'] -= data.bg
    if hasattr(data, 'medflux'):
        data['medflux'] -= np.median(data.bg, axis=0)

    if ('uncal' not in meta.suffix and meta.bg_dir == 'CxC' and
            meta.inst != 'wfc3'):
        # Save BG value at source position and BG stddev (no outlier rejection)
        coords = list(data.coords.keys())
        coords.remove('y')
        data['skylev'] = (coords, np.zeros_like(data.flux[:, 0]))
        data['skylev'].attrs['flux_units'] = data['flux'].attrs['flux_units']
        bg_inds = np.ma.getdata(deepcopy(data.mask))
        if meta.orders is None:
            data['skylev'] = data.bg[:, meta.src_ypos, :]
            bg_inds[:, meta.bg_y1:meta.bg_y2, :] = True
        else:
            for k in range(len(meta.orders)):
                data['skylev'][:, :, k] = data.bg[:, meta.src_ypos[k], :, k]
                bg_inds[:, meta.bg_y1[k]:meta.bg_y2[k], :, k] = True
        bg_data = np.ma.masked_where(bg_inds, data.flux, copy=True)
        bg_data = (np.ma.std(bg_data, axis=1))/np.sqrt(np.sum(~bg_inds))
        data['skyerr'] = (coords, bg_data)
        data['skyerr'].attrs['flux_units'] = data['flux'].attrs['flux_units']

    # Make image+background plots
    if isplots >= 3:
        if meta.orders is None:
            plots_s3.image_and_background(data, meta, log, m, group=group)
        else:
            for order in meta.orders:
                plots_s3.image_and_background(data.sel(order=order), meta,
                                              log, m, order=order, group=group)

    return data


@plots.apply_style
def fitbg(dataim, meta, mask, x1, x2, deg=1, threshold=5, isrotate=0,
          isplots=0):
    '''Fit sky background with out-of-spectra data.

    Parameters
    ----------
    dataim : ndarray
        The data array.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    mask : ndarray
        A boolean mask array, where True values are masked.
    x1 : ndarray
    x2 : ndarray
    deg : int; optional
        Polynomial order for column-by-column background subtraction
        Default is 1.
    threshold : int; optional
        Sigma threshold for outlier rejection during background subtraction.
        Defaullt is 5.
    isrotate : int; optional
        Default is 0 (no rotation).
    isplots : int; optional
        The amount of plots saved; set in ecf. Default is 0.
    '''
    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask = mask[::-1].T
    elif isrotate == 2:
        dataim = dataim.T
        mask = mask.T

    # Convert x1 and x2 to array, if need be
    ny, nx = np.shape(dataim)
    if isinstance(x1, (int, np.int32, np.int64)):
        x1 = np.zeros(ny, dtype=int)+x1
    if isinstance(x2, (int, np.int32, np.int64)):
        x2 = np.zeros(ny, dtype=int)+x2

    if deg < 0:
        # Calculate median background of entire frame
        # Assumes all x1 and x2 values are the same
        submask = np.concatenate((mask[:, :x1[0]].T, mask[:, x2[0]+1:].T)).T
        subdata = np.concatenate((dataim[:, :x1[0]].T,
                                  dataim[:, x2[0]+1:].T)).T
        bg = np.zeros((ny, nx)) + np.median(subdata[submask])
    elif deg is None:
        # No background subtraction
        bg = np.zeros((ny, nx))
    else:
        degs = np.ones(ny)*deg
        # Initiate background image with zeros
        bg = np.zeros((ny, nx))
        # Fit polynomial to each column
        for j in range(ny):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals = np.concatenate((range(x1[j]),
                                    range(x2[j]+1, nx))).astype(int)
            # If too few good pixels then average
            too_few_pix = (np.sum(~mask[j, :x1[j]]) < deg
                           or np.sum(~mask[j, x2[j]+1:]) < deg)
            if too_few_pix:
                degs[j] = 0
            while not nobadpixels:
                goodxvals = xvals[~mask[j, xvals]]
                dataslice = dataim[j, goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    nobadpixels = True  # exit while loop
                    # Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of
                    # degree 'deg'
                    coeffs = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model = np.polyval(coeffs, goodxvals)
                    # Calculate residuals and number of sigma from the model
                    residuals = dataslice - model
                    # Choose method for finding bad pixels
                    if meta.bg_method == 'median':
                        # Median Absolute Deviation (slower but more robust)
                        stdres = np.median(np.abs(np.ediff1d(residuals)))
                    elif meta.bg_method == 'mean':
                        # Mean Absolute Deviation (good compromise)
                        stdres = np.mean(np.abs(np.ediff1d(residuals)))
                    else:
                        # Simple standard deviation (faster but prone to
                        # missing scanned background stars)
                        # Default to standard deviation with no input
                        stdres = np.std(residuals)
                    if stdres == 0:
                        stdres = np.inf
                    stdevs = np.abs(residuals) / stdres
                    # Find worst data point
                    loc = np.argmax(stdevs)
                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        mask[j, goodxvals[loc]] = True
                    else:
                        nobadpixels = True  # exit while loop
            # Evaluate background model at all points, write model to
            # background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))
                if isplots == 6:
                    plt.figure(3601)
                    plt.clf()
                    plt.title(str(j))
                    plt.plot(goodxvals, dataslice, 'bo')
                    plt.plot(range(nx), bg[j], 'g-')
                    fname = ('figs'+os.sep+'Fig3601_BG_'+str(j) +
                             plots.get_filetype())
                    plt.savefig(meta.outputdir + fname, dpi=300)
                    plt.pause(0.01)

    if isrotate == 1:
        bg = (bg.T)[::-1]
        mask = (mask.T)[::-1]
    elif isrotate == 2:
        bg = (bg.T)
        mask = (mask.T)

    return bg, mask


@plots.apply_style
def fitbg2(dataim, meta, mask, bgmask, deg=1, threshold=5, isrotate=0,
           isplots=0):
    '''Fit sky background with out-of-spectra data.

    fitbg2 uses bgmask, a mask for the background region which enables fitting
    more complex background regions than simply above or below a given distance
    from the trace. This will help mask the 2nd and 3rd orders of NIRISS.

    Parameters
    ----------
    dataim : ndarray
        The data array.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    mask : ndarray
        A boolean mask array, where True values are masked.
    bgmask : ndarray
        A boolean background mask array, where True values are not part of the
        background.
    deg : int; optional
        Polynomial order for column-by-column background subtraction.
        Default is 1.
    threshold : int; optional
        Sigma threshold for outlier rejection during background subtraction.
        Default is 5.
    isrotate : int; optional
        Default is 0 (no rotation).
    isplots : int; optional
        The amount of plots saved; set in ecf. Default is 0.
    '''
    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask = mask[::-1].T
        bgmask = bgmask[::-1].T

    elif isrotate == 2:
        dataim = dataim.T
        mask = mask.T
        bgmask = bgmask.T

    # Initiate background image with zeros
    ny, nx = np.shape(dataim)
    bg = np.zeros((ny, nx))
    mask2 = mask | bgmask
    if deg < 0:
        # Calculate median background of entire frame
        bg += np.median(dataim[~mask2])

    elif deg is None:
        # No background subtraction
        pass
    else:
        degs = np.ones(ny)*deg
        # Fit polynomial to each column
        for j in tqdm(range(ny)):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals = np.where(~bgmask[j])[0]
            # If too few good pixels on either half of detector then
            # compute average
            too_few_pixels = (np.sum(~bgmask[j, :int(nx/2)]) < deg
                              or np.sum(~bgmask[j, int(nx/2):nx]) < deg)
            if too_few_pixels:
                degs[j] = 0
            while not nobadpixels:
                goodxvals = xvals[~bgmask[j, xvals]]
                dataslice = dataim[j, goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    nobadpixels = True  # exit while loop
                    # Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of
                    # degree 'deg'
                    coeffs = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model = np.polyval(coeffs, goodxvals)

                    # model = smooth.smooth(dataslice, window_len=window_len,
                    #                       window=windowtype)
                    # model = sps.medfilt(dataslice, window_len)
                    if isplots == 6:
                        plt.figure(3601)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        fname = ('figs'+os.sep+'Fig6_BG_'+str(j) +
                                 plots.get_filetype())
                        plt.savefig(meta.outputdir + fname, dpi=300)
                        plt.pause(0.01)

                    # Calculate residuals
                    residuals = dataslice - model

                    # Find worst data point
                    loc = np.argmax(np.abs(residuals))
                    # Calculate standard deviation of points excluding
                    # worst point
                    ind = np.arange(0, len(residuals), 1)
                    ind = np.delete(ind, loc)
                    stdres = np.std(residuals[ind])

                    if stdres == 0:
                        stdres = np.inf
                    # Calculate number of sigma from the model
                    stdevs = np.abs(residuals) / stdres

                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        bgmask[j, goodxvals[loc]] = True
                    else:
                        nobadpixels = True  # exit while loop

                    if isplots == 6:
                        plt.figure(3601)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        plt.pause(0.01)
                        plt.show()

            # Evaluate background model at all points, write model
            # to background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))

    if isrotate == 1:
        bg = (bg.T)[::-1]
        mask = (mask.T)[::-1]
        bgmask = (bgmask.T)[::-1]
    elif isrotate == 2:
        bg = (bg.T)
        mask = (mask.T)
        bgmask = (bgmask.T)

    return bg, bgmask
