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


def genlc(eventlabel, ecf_path='./', s3_meta=None):
    '''Compute photometric flux over specified range of wavelengths.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    ecf_path:   str
        The absolute or relative path to where ecfs are stored
    s3_meta:    MetaClass
        The metadata object from Eureka!'s S3 step (if running S3 and S4 sequentially).

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

    if s3_meta == None:
        #load savefile
        s3_meta = read_s3_meta(meta)

    meta = load_general_s3_meta_info(meta, ecf_path, s3_meta)

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider one aperture
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    # Create directories for Stage 5 outputs
    meta.runs_s4 = []
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            run = util.makedirectory(meta, 'S4', ap=spec_hw_val, bg=bg_hw_val)
            meta.runs_s4.append(run)

    run_i = 0
    old_meta = meta
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta = load_specific_s3_meta_info(old_meta, ecf_path, run_i, spec_hw_val, bg_hw_val)

            # Get directory for Stage 4 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S4', meta.runs_s4[run_i], ap=spec_hw_val, bg=bg_hw_val)
            run_i += 1

            # Copy existing S3 log file and resume log
            meta.s4_logname  = meta.outputdir + 'S4_' + meta.eventlabel + ".log"
            log         = logedit.Logedit(meta.s4_logname, read=meta.s3_logname)
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

def read_s3_meta(meta):

    # Search for the S2 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    fnames = glob.glob(rootdir+'S3_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(fnames)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        fnames = glob.glob(rootdir+'**/S3_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)
        fnames = sn.sort_nicely(fnames)

    if len(fnames)>=1:
        # get the folder with the latest modified time
        fname = max(fnames, key=os.path.getmtime)

    if len(fnames)==0:
        # There may be no metafiles in the inputdir - raise an error and give a helpful message
        raise AssertionError('Unable to find an output metadata file from Eureka!\'s S3 step '
                            +'in the inputdir: \n"{}"!'.format(rootdir))
    elif len(fnames)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(fname)
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = fname[:-4] # Strip off the .dat ending

    s3_meta = me.loadevent(fname)

    # Code to not break backwards compatibility with old MetaClass save files but also use the new MetaClass going forwards
    s3_meta = readECF.MetaClass(**s3_meta.__dict__)

    return s3_meta

def load_general_s3_meta_info(meta, ecf_path, s3_meta):
    # Need to remove the topdir from the outputdir
    s3_outputdir = s3_meta.outputdir[len(meta.topdir):]
    if s3_outputdir[0]=='/':
        s3_outputdir = s3_outputdir[1:]
    if s3_outputdir[-1]!='/':
        s3_outputdir += '/'

    meta = s3_meta

    # Load S4 Eureka! control file and store values in the S3 metadata object
    ecffile = 'S4_' + meta.eventlabel + '.ecf'
    meta.read(ecf_path, ecffile)

    # Overwrite the inputdir with the exact output directory from S3
    meta.inputdir = s3_outputdir
    meta.old_datetime = meta.datetime  # Capture the date that the S3 data was made (to figure out it's foldername)
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir

    return meta

def load_specific_s3_meta_info(meta, ecf_path, run_i, spec_hw_val, bg_hw_val):
    # Do some folder swapping to be able to reuse this function to find the correct S3 outputs
    tempfolder = meta.outputdir_raw
    meta.outputdir_raw = '/'.join(meta.inputdir_raw.split('/')[:-2])
    meta.inputdir = util.pathdirectory(meta, 'S3', meta.runs[run_i], old_datetime=meta.old_datetime, ap=spec_hw_val, bg=bg_hw_val)
    meta.outputdir_raw = tempfolder

    # Read in the correct S3 metadata for this aperture pair
    tempfolder = meta.inputdir
    meta.inputdir = meta.inputdir[len(meta.topdir):]
    new_meta = read_s3_meta(meta)
    meta.inputdir = tempfolder

    # Load S4 Eureka! control file and store values in the S3 metadata object
    ecffile = 'S4_' + meta.eventlabel + '.ecf'
    new_meta.read(ecf_path, ecffile)

    # Save correctly identified folders from earlier
    new_meta.inputdir = meta.inputdir
    new_meta.outputdir = meta.outputdir
    new_meta.inputdir_raw = meta.inputdir_raw
    new_meta.outputdir_raw = meta.outputdir_raw

    new_meta.runs_s4 = meta.runs_s4
    new_meta.datetime = meta.datetime

    new_meta.spec_hw = spec_hw_val
    new_meta.bg_hw = bg_hw_val

    return new_meta
