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

import os
import time as time_pkg
import numpy as np
import scipy.interpolate as spi
import astraeus.xarrayIO as xrio
from astropy.convolution import Box1DKernel
from . import plots_s4, drift, generate_LD, wfc3
from ..lib import logedit
from ..lib import readECF
from ..lib import manageevent as me
from ..lib import util
from ..lib import clipping


def genlc(eventlabel, ecf_path=None, s3_meta=None):
    '''Compute photometric flux over specified range of wavelengths.
    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
        Defaults to None which resolves to './'.
    s3_meta : eureka.lib.readECF.MetaClass
        The metadata object from Eureka!'s S3 step (if running S3 and S4
        sequentially). Defaults to None.
    Returns
    -------
    spec : Astreaus object 
        Data object of wavelength-like arrrays.
    lc : Astreaus object 
        Data object of time-like arrrays (light curve).
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S4.
    Notes
    -----
    History:
    - June 2021 Kevin Stevenson
        Initial version
    - October 2021 Taylor Bell
        Updated to allow for inputs from new S3
    - April 2022 Kevin Stevenson
        Enabled Astraeus
    - July 2022 Caroline Piaulet
        Recording of x (computed in S4) and y (computed in S3) pos drifts and 
        widths in Spec and LC objects
    '''
    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel
    meta.datetime = time_pkg.strftime('%Y-%m-%d')

    if s3_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into
        # that old MetaClass
        s3_meta, meta.inputdir, meta.inputdir_raw = \
            me.findevent(meta, 'S3', allowFail=False)
    else:
        # Running these stages sequentially, so can safely assume
        # the path hasn't changed
        meta.inputdir = s3_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]

    meta = me.mergeevents(meta, s3_meta)

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider
        # one aperture
        meta.spec_hw_range = [meta.spec_hw, ]
        meta.bg_hw_range = [meta.bg_hw, ]

    # Create directories for Stage 5 outputs
    meta.run_s4 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            meta.run_s4 = util.makedirectory(meta, 'S4', meta.run_s4,
                                             ap=spec_hw_val, bg=bg_hw_val)

    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            # Load in the S3 metadata used for this particular aperture pair
            meta = load_specific_s3_meta_info(meta)

            # Get directory for Stage 4 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S4', meta.run_s4,
                                                ap=meta.spec_hw, bg=meta.bg_hw)

            # Copy existing S3 log file and resume log
            meta.s4_logname = meta.outputdir + 'S4_' + meta.eventlabel + ".log"
            log = logedit.Logedit(meta.s4_logname, read=meta.s3_logname)
            log.writelog("\nStarting Stage 4: Generate Light Curves\n")
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf
            log.writelog('Copying S4 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            log.writelog(f"Loading S3 save file:\n{meta.filename_S3_SpecData}",
                         mute=(not meta.verbose))
            spec = xrio.readXR(meta.filename_S3_SpecData)

            if meta.wave_min is None:
                meta.wave_min = np.min(spec.wave_1d.values)
                log.writelog(f'No value was provided for meta.wave_min, so '
                             f'defaulting to {meta.wave_min}.',
                             mute=(not meta.verbose))
            elif meta.wave_min < np.min(spec.wave_1d.values):
                log.writelog(f'WARNING: The selected meta.wave_min '
                             f'({meta.wave_min}) is smaller than the shortest '
                             f'wavelength ({np.min(spec.wave_1d.values)})')
            if meta.wave_max is None:
                meta.wave_max = np.max(spec.wave_1d.values)
                log.writelog(f'No value was provided for meta.wave_max, so '
                             f'defaulting to {meta.wave_max}.',
                             mute=(not meta.verbose))
            elif meta.wave_max > np.max(spec.wave_1d.values):
                log.writelog(f'WARNING: The selected meta.wave_max '
                             f'({meta.wave_max}) is larger than the longest '
                             f'wavelength ({np.max(spec.wave_1d.values)})')

            meta.n_int, meta.subnx = spec.optspec.shape

            # Determine wavelength bins
            if not hasattr(meta, 'wave_hi'):
                binsize = (meta.wave_max - meta.wave_min)/meta.nspecchan
                meta.wave_low = np.round(np.linspace(meta.wave_min,
                                                     meta.wave_max-binsize,
                                                     meta.nspecchan), 3)
                meta.wave_hi = np.round(np.linspace(meta.wave_min+binsize,
                                                    meta.wave_max,
                                                    meta.nspecchan), 3)
            elif (meta.nspecchan is not None
                  and meta.nspecchan != len(meta.wave_hi)):
                log.writelog(f'WARNING: Your nspecchan value of '
                             f'{meta.nspecchan} differs from the size of '
                             f'wave_hi ({len(meta.wave_hi)}). Using the '
                             f'latter instead.')
                meta.nspecchan = len(meta.wave_hi)
            meta.wave_low = np.array(meta.wave_low)
            meta.wave_hi = np.array(meta.wave_hi)
            meta.wave = (meta.wave_low + meta.wave_hi)/2

            # Define light curve DataArray
            lcdata = xrio.makeLCDA(np.zeros((meta.nspecchan, meta.n_int)),
                                   meta.wave, spec.time.values,
                                   spec.optspec.attrs['flux_units'],
                                   spec.wave_1d.attrs['wave_units'],
                                   spec.optspec.attrs['time_units'],
                                   name='data')
            lcerr = xrio.makeLCDA(np.zeros((meta.nspecchan, meta.n_int)),
                                  meta.wave, spec.time.values,
                                  spec.optspec.attrs['flux_units'],
                                  spec.wave_1d.attrs['wave_units'],
                                  spec.optspec.attrs['time_units'],
                                  name='err')
            lcmask = xrio.makeLCDA(np.zeros((meta.nspecchan, meta.n_int),
                                            dtype=bool),
                                   meta.wave, spec.time.values, 'None',
                                   spec.wave_1d.attrs['wave_units'],
                                   spec.optspec.attrs['time_units'],
                                   name='mask')
            lc = xrio.makeDataset({'data': lcdata, 'err': lcerr,
                                   'mask': lcmask})
            if hasattr(spec, 'scandir'):
                lc['scandir'] = spec.scandir
            if hasattr(spec, 'drift2D'):
                lc['drift2D'] = spec.drift2D
            lc['wave_low'] = (['wavelength'], meta.wave_low)
            lc['wave_hi'] = (['wavelength'], meta.wave_hi)
            lc['wave_mid'] = (lc.wave_hi + lc.wave_low)/2
            lc['wave_err'] = (lc.wave_hi - lc.wave_low)/2
            lc.wave_low.attrs['wave_units'] = spec.wave_1d.attrs['wave_units']
            lc.wave_hi.attrs['wave_units'] = spec.wave_1d.attrs['wave_units']
            lc.wave_mid.attrs['wave_units'] = spec.wave_1d.attrs['wave_units']
            lc.wave_err.attrs['wave_units'] = spec.wave_1d.attrs['wave_units']

            if not hasattr(meta, 'boundary'):
                # The default value before this was added as an option
                meta.boundary = 'extend'

            # Do 1D sigma clipping (along time axis) on unbinned spectra
            if meta.sigma_clip:
                log.writelog('Sigma clipping unbinned optimal spectra along '
                             'time axis...')
                outliers = 0
                for w in range(meta.subnx):
                    spec.optspec[:, w], spec.optmask[:, w], nout = \
                        clipping.clip_outliers(spec.optspec[:, w], log,
                                               spec.wave_1d[w].values,
                                               spec.wave_1d.wave_units,
                                               mask=spec.optmask[:, w],
                                               sigma=meta.sigma,
                                               box_width=meta.box_width,
                                               maxiters=meta.maxiters,
                                               boundary=meta.boundary,
                                               fill_value=meta.fill_value,
                                               verbose=meta.verbose)
                    outliers += nout
                # Print summary if not verbose
                log.writelog(f'Identified a total of {outliers} outliers in '
                             f'time series, or an average of '
                             f'{outliers/meta.subnx:.3f} outliers per '
                             f'wavelength',
                             mute=meta.verbose)

            if hasattr(meta, 'record_ypos') and meta.record_ypos:
                lc['driftypos'] = (['time'], spec.driftypos.data)
                lc['driftywidth'] = (['time'], spec.driftywidth.data)
            
            # Record and correct for 1D drift/jitter
            if meta.recordDrift or meta.correctDrift:
                # Calculate drift over all frames and non-destructive reads
                # This can take a long time, so always print this message
                log.writelog('Computing drift/jitter')
                # Compute drift/jitter
                drift_results = drift.spec1D(spec.optspec, meta, log,
                                             mask=spec.optmask)
                drift1d, driftwidth, driftmask = drift_results
                # Replace masked points with moving mean
                drift1d = clipping.replace_moving_mean(
                    drift1d, driftmask, Box1DKernel(meta.box_width))
                driftwidth = clipping.replace_moving_mean(
                    driftwidth, driftmask, Box1DKernel(meta.box_width))
                lc['driftxpos'] = (['time'], drift1d)
                lc['driftxwidth'] = (['time'], driftwidth)
                lc['driftmask'] = (['time'], driftmask)
                
                spec['driftxpos'] = (['time'], drift1d)
                spec['driftxwidth'] = (['time'], driftwidth)
                spec['driftmask'] = (['time'], driftmask)
                
                if meta.correctDrift:
                    log.writelog('Applying drift/jitter correction')

                    # Correct for drift/jitter
                    for n in range(meta.n_int):
                        # Need to zero-out the weights of masked data
                        weights = (~spec.optmask[n]).astype(int)
                        spline = spi.UnivariateSpline(np.arange(meta.subnx),
                                                      spec.optspec[n], k=3,
                                                      s=0, w=weights)
                        spline2 = spi.UnivariateSpline(np.arange(meta.subnx),
                                                       spec.opterr[n], k=3,
                                                       s=0, w=weights)
                        optmask = spec.optmask[n].astype(float)
                        spline3 = spi.UnivariateSpline(np.arange(meta.subnx),
                                                       optmask, k=3, s=0,
                                                       w=weights)
                        spec.optspec[n] = spline(np.arange(meta.subnx) +
                                                 lc.driftxpos[n].values)
                        spec.opterr[n] = spline2(np.arange(meta.subnx) +
                                                 lc.driftxpos[n].values)
                        # Also shift mask if moving by >= 0.5 pixels
                        optmask = spline3(np.arange(meta.subnx) +
                                          lc.driftxpos[n].values)
                        spec.optmask[n] = optmask >= 0.5
                # Plot Drift
                if meta.isplots_S4 >= 1:
                    plots_s4.driftxpos(meta, lc)
                    plots_s4.driftxwidth(meta, lc)

            if hasattr(meta, 'sum_reads') and meta.sum_reads:
                # Sum each read from a scan together
                spec, lc, meta = wfc3.sum_reads(spec, lc, meta)

            # Compute MAD value
            meta.mad_s4 = util.get_mad(meta, log, spec.wave_1d.values,
                                       spec.optspec, spec.optmask,
                                       meta.wave_min, meta.wave_max)
            log.writelog(f"Stage 4 MAD = {np.round(meta.mad_s4, 2):.2f} ppm")

            if meta.isplots_S4 >= 1:
                plots_s4.lc_driftcorr(meta, spec.wave_1d, spec.optspec,
                                      optmask=spec.optmask)

            log.writelog("Generating light curves")

            # Loop over spectroscopic channels
            meta.mad_s4_binned = []
            for i in range(meta.nspecchan):
                log.writelog(f"  Bandpass {i} = {lc.wave_low.values[i]:.3f} - "
                             f"{lc.wave_hi.values[i]:.3f}")
                # Compute valid indeces within wavelength range
                index = np.where((spec.wave_1d >= lc.wave_low.values[i]) *
                                 (spec.wave_1d < lc.wave_hi.values[i]))[0]
                # Make masked arrays for easy summing
                optspec_ma = np.ma.masked_where(spec.optmask[:, index],
                                                spec.optspec[:, index])
                opterr_ma = np.ma.masked_where(spec.optmask[:, index],
                                               spec.opterr[:, index])
                # Compute mean flux for each spectroscopic channel
                # Sumation leads to outliers when there are masked points
                lc['data'][i] = np.ma.mean(optspec_ma, axis=1)
                # Add uncertainties in quadrature
                # then divide by number of good points to get
                # proper uncertainties
                lc['err'][i] = (np.sqrt(np.ma.sum(opterr_ma**2, axis=1)) /
                                np.ma.MaskedArray.count(opterr_ma, axis=1))

                # Do 1D sigma clipping (along time axis) on binned spectra
                if meta.sigma_clip:
                    lc['data'][i], lc['mask'][i], nout = \
                        clipping.clip_outliers(
                            lc.data[i], log, lc.data.wavelength[i].values,
                            lc.data.wave_units, mask=lc.mask[i],
                            sigma=meta.sigma, box_width=meta.box_width,
                            maxiters=meta.maxiters, boundary=meta.boundary,
                            fill_value=meta.fill_value, verbose=False)
                    log.writelog(f'  Sigma clipped {nout} outliers in time'
                                 f' series', mute=(not meta.verbose))

                # Plot each spectroscopic light curve
                if meta.isplots_S4 >= 3:
                    plots_s4.binned_lightcurve(meta, log, lc, i)

            # If requested, also generate white-light light curve
            if hasattr(meta, 'compute_white') and meta.compute_white:
                log.writelog("Generating white-light light curve")

                # Compute valid indeces within wavelength range
                index = np.where((spec.wave_1d >= meta.wave_min) *
                                 (spec.wave_1d < meta.wave_max))[0]
                central_wavelength = np.mean(spec.wave_1d[index].values)
                lc['flux_white'] = xrio.makeTimeLikeDA(np.zeros(meta.n_int),
                                                       lc.time,
                                                       lc.data.flux_units,
                                                       lc.time.time_units,
                                                       'flux_white')
                lc['err_white'] = xrio.makeTimeLikeDA(np.zeros(meta.n_int),
                                                      lc.time,
                                                      lc.data.flux_units,
                                                      lc.time.time_units,
                                                      'err_white')
                lc['mask_white'] = xrio.makeTimeLikeDA(np.zeros(meta.n_int,
                                                                dtype=bool),
                                                       lc.time, 'None',
                                                       lc.time.time_units,
                                                       'mask_white')
                lc.flux_white.attrs['wavelength'] = central_wavelength
                lc.flux_white.attrs['wave_units'] = lc.data.wave_units
                lc.err_white.attrs['wavelength'] = central_wavelength
                lc.err_white.attrs['wave_units'] = lc.data.wave_units
                lc.mask_white.attrs['wavelength'] = central_wavelength
                lc.mask_white.attrs['wave_units'] = lc.data.wave_units
                
                log.writelog(f"  White-light Bandpass = {meta.wave_min:.3f} - "
                             f"{meta.wave_max:.3f}")
                # Make masked arrays for easy summing
                optspec_ma = np.ma.masked_where(spec.optmask.values[:, index],
                                                spec.optspec.values[:, index])
                opterr_ma = np.ma.masked_where(spec.optmask.values[:, index],
                                               spec.opterr.values[:, index])
                # Compute mean flux for each spectroscopic channel
                # Sumation leads to outliers when there are masked points
                lc.flux_white[:] = np.ma.mean(optspec_ma, axis=1).data
                # Add uncertainties in quadrature
                # then divide by number of good points to get
                # proper uncertainties
                lc.err_white[:] = (np.sqrt(np.ma.sum(opterr_ma**2,
                                                     axis=1)) /
                                   np.ma.MaskedArray.count(opterr_ma,
                                                           axis=1)).data
                lc.mask_white[:] = np.ma.getmaskarray(np.ma.mean(optspec_ma,
                                                                 axis=1))

                # Do 1D sigma clipping (along time axis) on binned spectra
                if meta.sigma_clip:
                    lc.flux_white[:], lc.mask_white[:], nout = \
                        clipping.clip_outliers(
                            lc.flux_white, log, lc.flux_white.wavelength,
                            lc.data.wave_units, mask=lc.mask_white,
                            sigma=meta.sigma, box_width=meta.box_width,
                            maxiters=meta.maxiters, boundary=meta.boundary,
                            fill_value=meta.fill_value, verbose=False)
                    log.writelog(f'  Sigma clipped {nout} outliers in time '
                                 f' series')

                # Plot the white-light light curve
                if meta.isplots_S4 >= 3:
                    plots_s4.binned_lightcurve(meta, log, lc, 0, white=True)

            # Generate limb-darkening coefficients
            if hasattr(meta, 'compute_ld') and meta.compute_ld:
                log.writelog("Generating limb-darkening coefficients...",
                             mute=(not meta.verbose))
                ld_lin, ld_quad, ld_3para, ld_4para = \
                    generate_LD.exotic_ld(meta, spec)
                lc['exotic-ld_lin'] = (['wavelength', 'exotic-ld_1'], ld_lin)
                lc['exotic-ld_quad'] = (['wavelength', 'exotic-ld_2'], ld_quad)
                lc['exotic-ld_nonlin_3para'] = (['wavelength', 'exotic-ld_3'],
                                                ld_3para)
                lc['exotic-ld_nonlin_4para'] = (['wavelength', 'exotic-ld_4'],
                                                ld_4para)

            log.writelog('Saving results...')

            event_ap_bg = (meta.eventlabel + "_ap" + str(spec_hw_val) + '_bg'
                           + str(bg_hw_val))
            # Save Dataset object containing time-series of 1D spectra
            meta.filename_S4_SpecData = (meta.outputdir + 'S4_' + event_ap_bg
                                         + "_SpecData.h5")
            xrio.writeXR(meta.filename_S4_SpecData, spec, verbose=True)
            
            # Save Dataset object containing binned light curves
            meta.filename_S4_LCData = (meta.outputdir + 'S4_' + event_ap_bg
                                       + "_LCData.h5")
            xrio.writeXR(meta.filename_S4_LCData, lc, verbose=True)

            # Save results
            fname = meta.outputdir+'S4_'+meta.eventlabel+"_Meta_Save"
            me.saveevent(meta, fname, save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return spec, lc, meta


def load_specific_s3_meta_info(meta):
    """Load the specific S3 MetaClass object used to make this aperture pair.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.

    Returns
    -------
    eureka.lib.readECF.MetaClass
        The current metadata object with values from the old MetaClass.
    """
    # Get directory containing S3 outputs for this aperture pair
    inputdir = os.sep.join(meta.inputdir.split(os.sep)[:-2]) + os.sep
    inputdir += f'ap{meta.spec_hw}_bg{meta.bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into
    # that old MetaClass
    meta.inputdir = inputdir
    s3_meta, meta.inputdir, meta.inputdir_raw = \
        me.findevent(meta, 'S3', allowFail=False)
    # Merge S4 meta into old S3 meta
    meta = me.mergeevents(meta, s3_meta)

    return meta