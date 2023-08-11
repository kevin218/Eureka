import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing as mp

from astropy.stats import sigma_clip
from astropy.convolution import Box1DKernel, convolve

from scipy.signal import lombscargle
from scipy.optimize import minimize

from ..lib import plots


def computePixelTimes(integ, grp, SUBSIZE2, SUBSIZE1, NGROUPS, TSAMPLE,
                      TGROUP):
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


def smoothData(tempdata, boundary='extend', box_width=100):
    kernel = Box1DKernel(box_width)

    # Compute the moving mean
    bound_val = np.ma.median(tempdata)  # Only used if boundary=='fill'
    smoothed_data = convolve(tempdata, kernel, boundary=boundary,
                             fill_value=bound_val)
    
    return smoothed_data


def computeBG(cleaned, smooth=True):
    bg = cleaned[:, 11:60]
    bg = sigma_clip(bg, sigma=3)
    bg = sigma_clip(bg, sigma=3, axis=1)
    bg = sigma_clip(bg, sigma=3, axis=0)
    bg = np.ma.median(bg, axis=1)
    if smooth:
        bg = smoothData(bg, box_width=10, boundary='wrap')
        bg = smoothData(bg, box_width=50, boundary='extend')
    bg = bg.reshape((-1, 1))
    
    return bg


def model390(p0, x):
    Phase = p0
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
    Phase = p0
    if -0.5 <= Phase < 0.5:
        return np.sum((y-model390(p0, x))**2)
    else:
        return 1e50


def remove_390(p0, x, y):
    res = minimize(minfunc, p0, args=(x, y), method='Powell')
    return res.x


def get_pgram(data, time):
    freqs = np.linspace(1, 5e3, 1000)
    x = np.ma.copy(time.flatten())
    y = np.ma.copy(data.flatten())

    maskedInds = np.where(y.mask)
    x = np.delete(x, maskedInds)
    y = np.delete(y, maskedInds)

    pgram = lombscargle(x, y, freqs*2*np.pi, normalize=True)
    
    return pgram, freqs


def run_integ(images, dq, integ, SUBSIZE2, SUBSIZE1, NGROUPS, TSAMPLE, TGROUP,
              meta, returnp1=False, prnt=False, plot=False):
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
    
        if plot:
            pgram, freqs = get_pgram(cleaned[grp], pixel_times[grp])
            pgrams.append(pgram)
    
    # Package the data for fitting/plotting
    x = pixel_times.flatten()
    y = cleaned.flatten()

    p0 = [0, ]
    # Fit the 390 Hz noise
    p1 = remove_390(p0, x, y)
    if prnt:
        print(p1)

    if plot and meta.m == 0 and integ < meta.nplots:
        # Show the fit to a small chunk of data
        plt.figure(1501)
        plt.clf()
        
        plt.plot(x, y, '.', label='Raw Data')
        plt.plot(x, model390(p1, x), label='Fitted Model')
        plt.xlim(x[0]+0.01, x[0]+0.02)
        ymin = np.ma.min(sigma_clip(y, sigma=5))
        ymax = np.ma.max(sigma_clip(y, sigma=5))
        plt.ylim(ymin*0.99, ymax*1.1)
        plt.legend(loc='best')

        file_number = str(meta.m).zfill(
            int(np.floor(np.log10(meta.num_data_files))+1))
        int_number = str(integ).zfill(
            int(np.floor(np.log10(meta.n_int))+1))
        fname = f'figs{os.sep}fig1501_file{file_number}_int{int_number}'
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
    
        if (plot and meta.m == 0 and integ < meta.nplots and
                grp < meta.nplots):
            # Demonstrate the improvement in the noise power spectrum
            if meta.grouplevel_bg:
                cleaned_no390_nostar[grp] -= bg
            pgram2, freqs = get_pgram(cleaned_no390_nostar[grp],
                                      pixel_times[grp])

            plt.figure(1502)
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
            fname = f'figs{os.sep}fig1502_file{file_number}_int{int_number}'
            fname += f'_grp{grp_number}_LS_Periodogram'+plots.figure_filetype
            plt.savefig(meta.outputdir+fname, dpi=300, bbox_inches='tight')
            if not meta.hide_plots:
                plt.pause(0.2)
    
    if returnp1:
        return cleaned_no390, p1, integ
    else:
        return cleaned_no390, integ


def run(input_model, log, meta):
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
                meta, plot=(meta.isplots_S1 >= 5)))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        jobs = [pool.apply_async(func=run_integ,
                                 args=(input_model.data[integ],
                                       input_model.groupdq[integ] % 2 == 1,
                                       integ, SUBSIZE2, SUBSIZE1, NGROUPS,
                                       TSAMPLE, TGROUP, meta),
                                 kwds={'plot': (meta.isplots_S1 >= 5)},
                                 callback=writeImage)
                for integ in range(input_model.data.shape[0])]
        pool.close()
        iterfn = jobs
        if meta.verbose:
            iterfn = tqdm(iterfn, desc='  Looping through integrations')
        for job in iterfn:
            job.get()

    return input_model
