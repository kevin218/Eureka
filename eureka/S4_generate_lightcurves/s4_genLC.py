#! /usr/bin/env python

# Generic Stage 4 light curve generation pipeline


# Proposed Steps
# -------- -----
# 1.  Read in Stage 3 data products
# 2.  Replace NaNs with zero
# 3.  Determine wavelength bins
# 4.  Increase resolution of spectra (optional)
# 5.  Smooth spectra (optional)
# 6.  Applying 1D drift correction
# 7.  Generate light curves
# 8.  Save Stage 4 data products
# 9.  Produce plots


import os, glob
import time as time_pkg
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from . import plots_s4, drift
from ..lib import sort_nicely as sn
from ..lib import logedit
from ..lib import readECF
from ..lib import manageevent as me
from ..lib import astropytable
from ..lib import util
from ..lib import clipping

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return


def genlc(eventlabel, ecf_path=None, s3_meta=None):
    '''Compute photometric flux over specified range of wavelengths.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str, optional
        The absolute or relative path to where ecfs are stored. Defaults to None which resolves to './'.
    s3_meta : MetaClass
        The metadata object from Eureka!'s S3 step (if running S3 and S4 sequentially). Defaults to None.

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S4.

    Notes
    -----
    History:

    - June 2021 Kevin Stevenson
        Initial version
    - October 2021 Taylor Bell
        Updated to allow for inputs from new S3
    '''
    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel

    if s3_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into that old MetaClass
        s3_meta, meta.inputdir, meta.inputdir_raw = me.findevent(meta, 'S3', allowFail=False)
    else:
        # Running these stages sequentially, so can safely assume the path hasn't changed
        meta.inputdir = s3_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]
    
    meta = me.mergeevents(meta, s3_meta)

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider one aperture
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    # Create directories for Stage 5 outputs
    meta.run_s4 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            meta.run_s4 = util.makedirectory(meta, 'S4', meta.run_s4, ap=spec_hw_val, bg=bg_hw_val)

    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            # Load in the S3 metadata used for this particular aperture pair
            meta = load_specific_s3_meta_info(meta)

            # Get directory for Stage 4 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S4', meta.run_s4, ap=meta.spec_hw, bg=meta.bg_hw)

            # Copy existing S3 log file and resume log
            meta.s4_logname = meta.outputdir + 'S4_' + meta.eventlabel + ".log"
            log = logedit.Logedit(meta.s4_logname, read=meta.s3_logname)
            log.writelog("\nStarting Stage 4: Generate Light Curves\n")
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf
            log.writelog('Copying S4 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            log.writelog("Loading S3 save file", mute=(not meta.verbose))
            table = astropytable.readtable(meta.tab_filename)

            # Reverse the reshaping which has been done when saving the astropy table
            optspec = np.reshape(table['optspec'].data, (-1, meta.subnx))
            opterr = np.reshape(table['opterr'].data, (-1, meta.subnx))
            wave_1d = table['wave_1d'].data[0:meta.subnx]
            meta.time = table['time'].data[::meta.subnx]

            if meta.wave_min is None:
                meta.wave_min = np.min(wave_1d)
                log.writelog(f'No value was provided for meta.wave_min, so defaulting to {meta.wave_min}.', mute=(not meta.verbose))
            elif meta.wave_min<np.min(wave_1d):
                log.writelog(f'WARNING: The selected meta.wave_min ({meta.wave_min}) is smaller than the shortest wavelength ({np.min(wave_1d)})')
            if meta.wave_max is None:
                meta.wave_max = np.max(wave_1d)
                log.writelog(f'No value was provided for meta.wave_max, so defaulting to {meta.wave_max}.', mute=(not meta.verbose))
            elif meta.wave_max>np.max(wave_1d):
                log.writelog(f'WARNING: The selected meta.wave_max ({meta.wave_max}) is larger than the longest wavelength ({np.max(wave_1d)})')

            #Replace NaNs with zero
            optspec[np.where(np.isnan(optspec))] = 0
            opterr[np.where(np.isnan(opterr))] = 0
            meta.n_int, meta.subnx   = optspec.shape

            # Determine wavelength bins
            if not hasattr(meta, 'wave_hi'):
                binsize     = (meta.wave_max - meta.wave_min)/meta.nspecchan
                meta.wave_low = np.round([i for i in np.linspace(meta.wave_min, meta.wave_max-binsize, meta.nspecchan)],3)
                meta.wave_hi  = np.round([i for i in np.linspace(meta.wave_min+binsize, meta.wave_max, meta.nspecchan)],3)
            elif meta.nspecchan is not None and meta.nspecchan!=len(meta.wave_hi):
                log.writelog(f'WARNING: Your nspecchan value of {meta.nspecchan} differs from the size of wave_hi ({len(meta.wave_hi)}). Using the latter instead.')
                meta.nspecchan = len(meta.wave_hi)
            meta.wave_hi = np.array(meta.wave_hi)
            meta.wave_low = np.array(meta.wave_low)

            if not hasattr(meta, 'boundary'):
                meta.boundary = 'extend' # The default value before this was added as an option

            # Do 1D sigma clipping (along time axis) on unbinned spectra
            optspec = np.ma.masked_array(optspec)
            if meta.sigma_clip:
                log.writelog('Sigma clipping unbinned spectral time series', mute=(not meta.verbose))
                outliers = 0
                for l in range(meta.subnx):
                    optspec[:,l], nout = clipping.clip_outliers(optspec[:,l], log, wave_1d[l], meta.sigma, meta.box_width, meta.maxiters, meta.boundary, meta.fill_value, verbose=meta.verbose)
                    outliers += nout
                log.writelog('Identified a total of {} outliers in time series, or an average of {} outliers per wavelength'.format(outliers, np.round(outliers/meta.subnx, 1)), mute=meta.verbose)

            # Apply 1D drift/jitter correction
            if meta.correctDrift:
                #Calculate drift over all frames and non-destructive reads
                log.writelog('Applying drift/jitter correction') # This can take a long time, so always print this message
                # Compute drift/jitter
                meta = drift.spec1D(optspec, meta, log)
                # Correct for drift/jitter
                for n in range(meta.n_int):
                    # Need to zero-out the weights of masked data
                    weights = (~np.ma.getmaskarray(optspec[n])).astype(int)
                    spline     = spi.UnivariateSpline(np.arange(meta.subnx), optspec[n], k=3, s=0, w=weights)
                    spline2    = spi.UnivariateSpline(np.arange(meta.subnx), opterr[n],  k=3, s=0, w=weights)
                    optspec[n] = np.ma.masked_invalid(spline(np.arange(meta.subnx)+meta.drift1d[n]))
                    opterr[n]  = np.ma.masked_invalid(spline2(np.arange(meta.subnx)+meta.drift1d[n]))
                # Plot Drift
                if meta.isplots_S4 >= 1:
                    plots_s4.drift1d(meta)

            # Compute MAD alue
            meta.mad_s4 = util.get_mad(meta, wave_1d, optspec, meta.wave_min, meta.wave_max)
            log.writelog(f"Stage 4 MAD = {str(np.round(meta.mad_s4, 2))} ppm")

            if meta.isplots_S4 >= 1:
                plots_s4.lc_driftcorr(meta, wave_1d, optspec)

            log.writelog("Generating light curves")
            meta.lcdata   = np.ma.zeros((meta.nspecchan, meta.n_int))
            meta.lcerr    = np.ma.zeros((meta.nspecchan, meta.n_int))
            # ev.eventname2 = ev.eventname
            for i in range(meta.nspecchan):
                log.writelog(f"  Bandpass {i} = %.3f - %.3f" % (meta.wave_low[i], meta.wave_hi[i]))
                # Compute valid indeces within wavelength range
                index   = np.where((wave_1d >= meta.wave_low[i])*(wave_1d < meta.wave_hi[i]))[0]
                # Sum flux for each spectroscopic channel
                meta.lcdata[i]    = np.ma.sum(optspec[:,index],axis=1)
                # Add uncertainties in quadrature
                meta.lcerr[i]     = np.ma.sqrt(np.ma.sum(opterr[:,index]**2,axis=1))

                # Do 1D sigma clipping (along time axis) on binned spectra
                if meta.sigma_clip:
                    meta.lcdata[i], outliers = clipping.clip_outliers(meta.lcdata[i], log, wave_1d[l], meta.sigma, meta.box_width, meta.maxiters, meta.boundary, meta.fill_value, verbose=False)
                    log.writelog('  Sigma clipped {} outliers in time series'.format(outliers))

                # Plot each spectroscopic light curve
                if meta.isplots_S4 >= 3:
                    plots_s4.binned_lightcurve(meta, i)

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.writelog('Saving results as astropy table')
            event_ap_bg = meta.eventlabel + "_ap" + str(spec_hw_val) + '_bg' + str(bg_hw_val)
            meta.tab_filename_s4 = meta.outputdir + 'S4_' + event_ap_bg + "_Table_Save.txt"
            wavelengths = np.mean(np.append(meta.wave_low.reshape(1,-1), meta.wave_hi.reshape(1,-1), axis=0), axis=0)
            wave_errs = (meta.wave_hi-meta.wave_low)/2
            astropytable.savetable_S4(meta.tab_filename_s4, meta.time, wavelengths, wave_errs, meta.lcdata, meta.lcerr)

            # Save results
            log.writelog('Saving results')
            me.saveevent(meta, meta.outputdir + 'S4_' + meta.eventlabel + "_Meta_Save", save=[])

            # if (isplots >= 1) and (correctDrift == True):
            #     # Plot Drift
            #     # Plots corrected 2D light curves

            log.closelog()

    return meta

def load_specific_s3_meta_info(meta):
    # Get directory containing S3 outputs for this aperture pair
    inputdir = os.sep.join(meta.inputdir.split(os.sep)[:-2]) + os.sep
    inputdir += f'ap{meta.spec_hw}_bg{meta.bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into that old MetaClass
    meta.inputdir = inputdir
    s3_meta, meta.inputdir, meta.inputdir_raw = me.findevent(meta, 'S3', allowFail=False)
    # Merge S4 meta into old S3 meta
    meta = me.mergeevents(meta, s3_meta)

    return meta
