#! /usr/bin/env python

# Generic Stage 4cal Calibrated Stellar Spectra pipeline

import numpy as np
import os
import time as time_pkg
from copy import deepcopy
import astraeus.xarrayIO as xrio
import scipy.interpolate as spi

from ..version import version
from ..lib import manageevent as me
from ..lib import util, logedit
from ..S3_data_reduction.sigrej import sigrej
from .s4cal_meta import S4cal_MetaClass
from .plots_s4cal import plot_whitelc, plot_stellarSpec


def medianCalSpec(eventlabel, ecf_path=None, s3_meta=None, input_meta=None):
    '''Generate median calibrated stellar spectra using in-occultation data
    and out-of-occultation baseline.  The outputs also include the standard
    deviation in time, which can reasonably be used as uncertainties.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
        Defaults to None which resolves to './'.):
    s3_meta : eureka.lib.readECF.MetaClass; optional
        The metadata object from Eureka!'s S3 step (if running S3 and S4cal
        sequentially). Defaults to None.
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S4cal.
    '''
    s3_meta = deepcopy(s3_meta)
    input_meta = deepcopy(input_meta)

    if input_meta is None:
        # Load Eureka! control file and store values in Event object
        ecffile = 'S4cal_' + eventlabel + '.ecf'
        meta = S4cal_MetaClass(ecf_path, ecffile)
    else:
        meta = S4cal_MetaClass(**input_meta.__dict__)

    meta.version = version
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

    meta = S4cal_MetaClass(**me.mergeevents(meta, s3_meta).__dict__)
    meta.set_defaults()

    # Create directories for Stage 4cal outputs
    meta.run_S4cal = util.makedirectory(meta, 'S4cal', None)
    # Get the directory for Stage 4cal processing outputs
    meta.outputdir = util.pathdirectory(meta, 'S4cal', meta.run_S4cal)

    # Copy existing S3 log file and resume log
    meta.s4cal_logname = meta.outputdir+'S4cal_'+meta.eventlabel+'.log'
    log = logedit.Logedit(meta.s4cal_logname, read=meta.s3_logname)
    log.writelog("\nStarting Stage 4cal: \n")
    log.writelog(f"Eureka! Version: {meta.version}", mute=True)
    log.writelog(f"Output directory: {meta.outputdir}")

    # Copy ecf
    log.writelog('Copying S4cal control file')
    meta.copy_ecf()

    # Load Stage 3 specData.h5 file
    specData_savefile = (
        meta.inputdir +
        meta.filename_S3_SpecData.split(os.path.sep)[-1])
    log.writelog(f"Loading S3 save file:\n{specData_savefile}",
                 mute=(not meta.verbose))
    spec = xrio.readXR(specData_savefile)
    wave = spec.wave_1d.data
    log.writelog(f"Time range: {np.min(spec.time.values)} " +
                 f"- {np.max(spec.time.values)}")

    # Flag outliers in time
    if meta.photometry:
        mask = sigrej(spec.aplev.values, meta.sigma_thresh)
        optspec = np.ma.masked_where(mask, spec.aplev.values)
        flux_units = spec.aplev.flux_units
        wave_units = spec.wave_1d.wave_units
        time_units = spec.aplev.time_units
    else:
        mask = sigrej(spec.optspec.values, meta.sigma_thresh,
                      mask=spec.optmask.values, axis=0)
        optspec = np.ma.masked_array(spec.optspec.values, mask)
        flux_units = spec.optspec.flux_units
        wave_units = spec.optspec.wave_units
        time_units = spec.optspec.time_units

    # Apply aperture correction
    optspec *= meta.apcorr

    if isinstance(meta.t0, float):
        meta.t0 = [meta.t0]
    num_ecl = len(meta.t0)
    fig = None
    ax = None
    batch = []
    for i in range(num_ecl):
        # Compute baseline/in-occultation median spectra
        t0 = meta.t0[i] + meta.time_offset
        p = meta.period
        rprs = meta.rprs
        ars = meta.ars
        cosi = np.cos(meta.inc*np.pi/180)

        # This code snippet will automatically make sure t0 is within
        # the current observation window
        nOrbits = (np.mean(spec.time.data)-t0) // p + 1
        t0 += nOrbits*p

        # total occultation duration
        if meta.t14 is None:
            meta.t14 = p/np.pi*np.arcsin(
                1/ars*np.sqrt(((1 + rprs)**2 - (ars*cosi)**2)/(1 - cosi**2)))
            if not np.isfinite(meta.t14):
                raise Exception("t14 is not finite. Check your system " +
                                "parameters.")
        # Full occultation duration
        if meta.t23 is None:
            meta.t23 = p/np.pi*np.arcsin(
                1/ars*np.sqrt(((1 - rprs)**2 - (ars*cosi)**2)/(1 - cosi**2)))
            if not np.isfinite(meta.t23):
                raise Exception("t23 is not finite. Check your system " +
                                "parameters or planet may be grazing.")
        # Indices for first through fourth contact
        t_arr = spec.time.values
        t14_half = meta.t14 / 2
        t23_half = meta.t23 / 2
        base_dur = meta.base_dur
        it1_matches = np.flatnonzero(t_arr > (t0 - t14_half))
        it1 = it1_matches[0] if it1_matches.size > 0 else 0
        it2_matches = np.flatnonzero(t_arr > (t0 - t23_half))
        it2 = it2_matches[0] if it2_matches.size > 0 else 0
        it3_matches = np.flatnonzero(t_arr > (t0 + t23_half))
        it3 = it3_matches[0] if it3_matches.size > 0 else 0
        it4_matches = np.flatnonzero(t_arr > (t0 + t14_half))
        it4 = it4_matches[0] if it4_matches.size > 0 else len(t_arr) - 1
        # Indices for beginning and end of baseline
        if meta.base_dur is None:
            it0 = 0
            it5 = len(t_arr) - 1
        else:
            it0_matches = np.flatnonzero(t_arr > (t0 - t14_half - base_dur))
            it0 = it0_matches[0] if it0_matches.size > 0 else 0

            it5_matches = np.flatnonzero(t_arr > (t0 + t14_half + base_dur))
            it5 = it5_matches[0] if it5_matches.size > 0 else len(t_arr) - 1
        meta.it = [it0, it1, it2, it3, it4, it5]

        if meta.isplots_S4cal >= 2:
            fig, ax = plot_whitelc(optspec, spec.time, meta, i, fig=fig, ax=ax)

        # Median baseline (i.e. out-of-occultation) flux
        spec_baseline = np.ma.median(np.ma.concatenate(
            (optspec[it0:it1], optspec[it4:it5])), axis=0)
        std_baseline = np.ma.std(np.ma.concatenate(
            (optspec[it0:it1], optspec[it4:it5])), axis=0)

        if not meta.photometry:
            # Mask outliers along wavelength axis
            tck = spi.splrep(wave, spec_baseline, w=1/std_baseline, k=3,
                             s=meta.smoothing)
            spline = spi.splev(wave, tck)
            mask = sigrej(spec_baseline - spline, meta.sigma_thresh)
            igood = np.where(~mask)[0]
        else:
            spec_baseline = np.array([spec_baseline, ])
            std_baseline = np.array([std_baseline, ])
            igood = np.zeros(1, dtype=int)
        # Create XArray data arrays
        base_flux = xrio.makeLCDA(spec_baseline[igood, np.newaxis],
                                  wave[igood], [t0],
                                  flux_units, wave_units, time_units,
                                  name='Median Baseline Flux')
        base_fstd = xrio.makeLCDA(std_baseline[igood, np.newaxis],
                                  wave[igood], [t0],
                                  flux_units, wave_units, time_units,
                                  name='Std. Dev. of Baseline Flux')
        denom = max(it5 - it4 + it1 - it0, 1)
        base_ferr = base_fstd.copy() / np.sqrt(denom)

        # Median in-occultation flux
        spec_t23 = np.ma.median(optspec[it2:it3+1], axis=0)
        std_t23 = np.ma.std(optspec[it2:it3+1], axis=0)
        if not meta.photometry:
            # Mask outliers along wavelength axis
            tck = spi.splrep(wave, spec_t23, w=1/std_t23, k=3)
            spline = spi.splev(wave, tck)
            mask = sigrej(spec_t23 - spline, meta.sigma_thresh)
            igood = np.where(~mask)[0]
        else:
            spec_t23 = np.array([spec_t23, ])
            std_t23 = np.array([std_t23, ])
            igood = np.zeros(1, dtype=int)
        # Create XArray data arrays
        ecl_flux = xrio.makeLCDA(spec_t23[igood, np.newaxis],
                                 wave[igood], [t0],
                                 flux_units, wave_units, time_units,
                                 name='Median In-Occultation Flux')
        ecl_fstd = xrio.makeLCDA(std_t23[igood, np.newaxis],
                                 wave[igood], [t0],
                                 flux_units, wave_units, time_units,
                                 name='Std. Dev. of In-Occultation Flux')
        denom = max(it3 - it2 + 1, 1)
        ecl_ferr = ecl_fstd.copy() / np.sqrt(denom)

        # Create XArray dataset
        ds = dict(base_flux=base_flux, base_fstd=base_fstd,
                  base_ferr=base_ferr, ecl_flux=ecl_flux,
                  ecl_fstd=ecl_fstd, ecl_ferr=ecl_ferr)
        ds = xrio.makeDataset(ds)
        batch.append(ds)

    # Concatenate datasets
    ds = xrio.concat(batch, dim="time")
    filename = meta.outputdir+'S4cal_'+meta.eventlabel + "_CalStellarSpec.h5"
    xrio.writeXR(filename, ds, verbose=True)

    if meta.isplots_S4cal >= 2:
        plot_stellarSpec(meta, ds)

    return meta, spec, ds

