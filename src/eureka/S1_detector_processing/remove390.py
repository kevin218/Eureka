import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing as mp

from astropy.stats import sigma_clip

from scipy.signal import lombscargle
from scipy.optimize import minimize

from ..lib import plots
from ..lib.smooth import medfilt


def computePixelTimes(integ, grp, SUBSIZE2, SUBSIZE1, NGROUPS, TSAMPLE,
                      TGROUP):
    """Compute the time stamps for every pixel in the group.

    Based on code sent to TJB by Michael E Ressler.

    Parameters
    ----------
    integ : int
        The current integration number.
    grp : int
        The current group number.
    SUBSIZE2 : int
        The subarray ysize.
    SUBSIZE1 : int
        The subarray xsize.
    NGROUPS : int
        The number of groups per integration.
    TSAMPLE : float
        The sample time.
    TGROUP : float
        The group time.

    Returns
    -------
    pixel_times : np.array
        The time stamp for every pixel in the group.
    """
    # SLITLESSPRISM Subarray
    nrows = SUBSIZE2  # px
    ncols = SUBSIZE1  # px
    ngrp = NGROUPS

    nreadouts = 4  # There are four readouts that are all simultaneous
    dt_fourPix = TSAMPLE*1e-6  # s
    dt_rowSep = dt_fourPix*11  # s (Not sure why it is 11 or if it is always)
    dt_oneRow = (ncols/nreadouts-1)*dt_fourPix + dt_rowSep  # s
    dt_grp = TGROUP  # s
    dt_int = dt_grp*(ngrp+1)  # s

    # Time steps within one row
    dt_cols = np.repeat(np.arange(ncols/nreadouts, dtype=float)*dt_fourPix,
                        nreadouts)

    # Time steps between rows
    dt_rows = np.arange(nrows, dtype=float)*dt_oneRow

    # Reshape and add together to get 2D time array
    dt_rows = dt_rows.reshape((-1, 1))
    dt_cols = dt_cols.reshape((1, -1))
    pixel_times = dt_cols+dt_rows+(dt_grp*grp)+(dt_int*integ)

    return pixel_times


def computeBG(cleaned, smooth=True):
    """Compute the background in the group-level data.

    This differs from the more general GLBS code written for other instruments
    since there are several MIRI-specific oddities that need to be handled.

    Parameters
    ----------
    cleaned : np.ma.MaskedArray
        A masked array containing all data for a single integration.
    smooth : bool; optional
        Whether or not boxcar smoothing should be applied when computing the
        background level. This is useful when trying to measure the 390Hz noise
        component which is largely but not entirely removed by row-by-row
        background subtraction. True by default.

    Returns
    -------
    bg : np.ma.MaskedArray
        A masked array containing the computed background of shape
        (cleaned.shape[0], 1).
    """
    bg = cleaned[:, 11:61]
    bg = sigma_clip(bg, sigma=3)
    bg = sigma_clip(bg, sigma=3, axis=1)
    bg = sigma_clip(bg, sigma=3, axis=0)
    bg = np.ma.median(bg, axis=1)
    if smooth:
        bg = medfilt(bg, 51)
    bg = bg.reshape((-1, 1))

    return bg


def model390(p0, x):
    """A function to model the 390Hz noise with an input phase.

    Parameters
    ----------
    p0 : float
        The phase offset of the 390Hz noise in units of revolutions (0--1).
    x : np.array
        The time stamps at which to compute the 390Hz noise signal.

    Returns
    -------
    np.array
        The predicted 390Hz noise signal as a function of time.
    """
    Phase = p0[0]
    Phase1 = 0.46983885+Phase
    Phase2 = 0.42408343+Phase
    Phase4 = 0.24882948+Phase
    Amp1 = 3.72570491
    Amp2 = 0.74752371
    Amp4 = 0.22299596
    Offset = -0.15818526573255495
    f = 390.625*2*np.pi
    return (Amp1*np.sin(x*f+Phase1*2*np.pi) +
            Amp2*np.sin(x*f*2+Phase2*2*np.pi) +
            Amp4*np.sin(x*f*4+Phase4*2*np.pi) +
            Offset)


def minfunc(p0, x, y):
    """A cost function to use while fitting the 390Hz noise signal.

    Parameters
    ----------
    p0 : float
        The phase offset of the 390Hz noise in units of revolutions (0--1).
    x : np.array
        The time stamps at which to compute the 390Hz noise signal.
    y : np.array
        The observed DN for each of the pixels to compare the model against.

    Returns
    -------
    float
        The sum of the squares of the differences between the data and the
        model.
    """
    Phase = p0[0]
    if -0.5 <= Phase < 0.5:
        return np.sum((y-model390(p0, x))**2)
    else:
        return 1e50


def get_pgram(data, time):
    """Compute the Lomb-Scargle periodogram to visualize 390Hz noise signal.

    Parameters
    ----------
    data : np.array
        The DN counts for every pixel in the group.
    time : np.array
        The time stamps for every pixel in the group.

    Returns
    -------
    pgram : np.array
        The computed values of the periodogram.
    freqs : np.array
        The frequencies at which the periodogram were computed.
    """
    freqs = np.linspace(1, 5e3, 1000)
    x = np.ma.copy(time.flatten())
    y = np.ma.copy(data.flatten())

    maskedInds = np.where(y.mask)
    x = np.delete(x, maskedInds)
    y = np.delete(y, maskedInds)

    pgram = lombscargle(x, y, freqs*2*np.pi, normalize=True)

    return pgram, freqs


def run_integ(images, dq, integ, SUBSIZE2, SUBSIZE1, NGROUPS, TSAMPLE, TGROUP,
              meta, returnp1=False, prnt=False, isplots_S1=1):
    """Run the 390Hz noise removal on a single integration.

    Parameters
    ----------
    images : np.array
        The DN counts for every pixel in the integration.
    dq : np.array
        The DQ array from the FITS file.
    integ : int
        The current integration number (needed for computing the pixel time
        stamps).
    SUBSIZE2 : int
        The subarray ysize.
    SUBSIZE1 : int
        The subarray xsize.
    NGROUPS : int
        The number of groups per integration.
    TSAMPLE : float
        The sample time.
    TGROUP : float
        The group time.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    returnp1 : bool; optional
        If True, return the fitted phase of the 390Hz signal (useful when
        debugging). False by default.
    prnt : bool; optional
        If True, print the fitted phase of the 390Hz signal (useful when
        debugging). False by default.
    isplots_S1 : int; optional
        Sets the plotting verbosity of the function. If >=3, plot some figures
        showing the fitting of the 390Hz noise signal and the changes to the
        L-S periodogram (useful when debugging). If <3, no plots are made.
        By default, set to 1.

    Returns
    -------
    cleaned_no390 : np.array
        The images data with the 390Hz signal removed. The background flux
        will also have been removed if meta.grouplevel_bg is True.
    p1 : float; optional
        The fitted phase of the 390Hz signal. Only returned if returnp1 is
        True.
    integ : int
        The current integration number (needed for updating the input array).
    """
    # Mask known bad pixels
    images1 = np.ma.copy(np.ma.masked_invalid(images))
    images1 = np.ma.masked_where(dq, images1)
    images1[:, :, 68:].mask = True
    images1[:, :, :4].mask = True

    # Mask out the star for when fitting the noise
    images2 = np.ma.copy(images1)
    images2[:, 50:, 30:44].mask = True

    # Get the change in flux since the last group
    cleaned = np.ma.masked_invalid(images2[1:]-images2[:-1])

    # Get the time for each pixel
    pixel_times = []
    for grp in range(1, images2.shape[0]):
        pixel_times.append(computePixelTimes(integ, grp, SUBSIZE2, SUBSIZE1,
                                             NGROUPS, TSAMPLE, TGROUP))
    pixel_times = np.array(pixel_times)

    # Subtract a smoothed background
    pgrams = []
    for grp in range(images2.shape[0]-1):
        bg = computeBG(cleaned[grp], smooth=True)
        cleaned[grp] -= bg

        if isplots_S1 >= 3:
            pgram, freqs = get_pgram(cleaned[grp], pixel_times[grp])
            pgrams.append(pgram)

    # Package the data for fitting/plotting
    x = pixel_times.flatten()
    y = cleaned.flatten()

    p0 = [0., ]
    # Fit the 390 Hz noise
    res = minimize(minfunc, p0, args=(x, y), method='Powell')
    p1 = res.x
    if prnt:
        print(p1)

    if isplots_S1 >= 3 and meta.m == 0 and integ < meta.nplots:
        # Show the fit to a small chunk of data
        plt.figure(1301)
        plt.clf()

        plt.plot(x, y, '.', label='Raw Data')
        plt.plot(x, model390(p1, x), label='Fitted Model')
        plt.xlim(x[0]+0.01, x[0]+0.02)
        ymin = np.ma.min(sigma_clip(y, sigma=5))
        ymax = np.ma.max(sigma_clip(y, sigma=5))
        plt.ylim(ymin*0.99, ymax*1.1)
        plt.legend(loc='best')
        plt.ylabel('Mean-Subtracted Pixel Counts (DN)')
        plt.xlabel('Pixel Time Stamp (s)')

        file_number = str(meta.m).zfill(
            int(np.floor(np.log10(meta.num_data_files))+1))
        int_number = str(integ).zfill(
            int(np.floor(np.log10(meta.n_int))+1))
        fname = f'figs{os.sep}fig1301_file{file_number}_int{int_number}'
        fname += '_390HzFit'
        fname += plots.figure_filetype
        plt.savefig(meta.outputdir+fname, dpi=300, bbox_inches='tight')
        if not meta.hide_plots:
            plt.pause(0.2)

    # Remove the 390 Hz noise
    cleaned_no390_nostar = np.ma.masked_invalid(images2[1:]-images2[:-1])
    cleaned_no390_nostar -= model390(p1, pixel_times
                                     ).reshape(cleaned_no390_nostar.shape)

    cleaned_no390 = np.ma.copy(images)
    for grp in range(images2.shape[0]):
        # Since groups are non-destructive, need to subtract from current and
        # all subsequent groups
        pixel_times_temp = computePixelTimes(integ, grp, SUBSIZE2, SUBSIZE1,
                                             NGROUPS, TSAMPLE, TGROUP)
        cleaned_no390[grp:] -= model390(p1, pixel_times_temp
                                        ).reshape(cleaned_no390.shape[1:])

    # Remove the background
    for grp in range(images2.shape[0]-1):
        bg = computeBG(cleaned_no390_nostar[grp], smooth=False)
        # Since groups are non-destructive, need to subtract from current and
        # all subsequent groups
        if meta.grouplevel_bg:
            cleaned_no390[grp+1:] -= bg

        if (isplots_S1 >= 3 and meta.m == 0 and integ < meta.nplots and
                grp < meta.nplots):
            # Demonstrate the improvement in the noise power spectrum
            if meta.grouplevel_bg:
                cleaned_no390_nostar[grp] -= bg
            pgram2, freqs = get_pgram(cleaned_no390_nostar[grp],
                                      pixel_times[grp])

            plt.figure(1302)
            plt.clf()

            plt.semilogy(freqs, pgrams[grp], c='r', label='Raw')
            plt.semilogy(freqs, pgram2, label='Cleaned')

            plt.axvline(390.625, c='r', alpha=0.2)
            plt.axvline(390.625*2, c='r', alpha=0.2)
            plt.axvline(390.625*3, c='k', alpha=0.2)
            plt.axvline(390.625*4, c='r', alpha=0.2)
            plt.axvline(1395, c='k', alpha=0.2)
            plt.axvline(2180, c='k', alpha=0.2)
            plt.axvline(3180, c='k', alpha=0.2)
            plt.axvline(3580, c='k', alpha=0.2)
            plt.axvline(3960, c='k', alpha=0.2)

            plt.xlim(np.min(freqs), np.max(freqs))

            plt.legend(loc='best')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (Abitrary Units)')

            file_number = str(meta.m).zfill(
                int(np.floor(np.log10(meta.num_data_files))+1))
            int_number = str(integ).zfill(
                int(np.floor(np.log10(meta.n_int))+1))
            grp_number = str(grp+1).zfill(
                int(np.floor(np.log10(images2.shape[0]-1))+1))
            fname = f'figs{os.sep}fig1302_file{file_number}_int{int_number}'
            fname += f'_grp{grp_number}_LS_Periodogram'+plots.figure_filetype
            plt.savefig(meta.outputdir+fname, dpi=300, bbox_inches='tight')
            if not meta.hide_plots:
                plt.pause(0.2)

    if returnp1:
        return cleaned_no390, p1, integ
    else:
        return cleaned_no390, integ


def run(input_model, log, meta):
    """Run the 390 Hz noise removal step for every integration in a file.

    Parameters
    ----------
    input_model : jwst.datamodels.RampModel
        The input file segment to be processed.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    input_model : jwst.datamodels.RampModel
        The input file segment with the 390Hz noise removed. The background
        flux will also have been removed if meta.grouplevel_bg is True.
    """
    SUBSIZE2 = input_model.meta.subarray.ysize
    SUBSIZE1 = input_model.meta.subarray.xsize
    NGROUPS = input_model.meta.exposure.ngroups
    TSAMPLE = input_model.meta.exposure.sample_time
    TGROUP = input_model.meta.exposure.group_time

    # Write background
    def writeImage(args):
        images, integ = args
        input_model.data[integ] = images
        return

    # Remove 390Hz noise for each integration
    log.writelog('Removing 390 Hz Noise...')
    if meta.ncpu == 1:
        # Only 1 CPU
        iterfn = range(input_model.data.shape[0])
        if meta.verbose:
            iterfn = tqdm(iterfn, desc='  Looping through integrations')
        for integ in iterfn:
            writeImage(run_integ(
                input_model.data[integ],
                input_model.groupdq[integ] % 2 == 1,
                integ, SUBSIZE2, SUBSIZE1, NGROUPS, TSAMPLE, TGROUP,
                meta, isplots_S1=meta.isplots_S1))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        jobs = [pool.apply_async(func=run_integ,
                                 args=(input_model.data[integ],
                                       input_model.groupdq[integ] % 2 == 1,
                                       integ, SUBSIZE2, SUBSIZE1, NGROUPS,
                                       TSAMPLE, TGROUP, meta),
                                 kwds={'isplots_S1': meta.isplots_S1},
                                 callback=writeImage)
                for integ in range(input_model.data.shape[0])]
        pool.close()
        iterfn = jobs
        if meta.verbose:
            iterfn = tqdm(iterfn, desc='  Looping through integrations')
        for job in iterfn:
            job.get()

    return input_model
