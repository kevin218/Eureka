#! /usr/bin/env python

# Generic Stage 4cal Calibrated Stellar Spectra pipeline

import numpy as np
import os
import time as time_pkg
from copy import deepcopy
import astraeus.xarrayIO as xrio
import scipy.interpolate as spi
import matplotlib.pyplot as plt

from eureka.version import version
from eureka.lib import manageevent as me
from eureka.lib import util, logedit, plots
from eureka.S3_data_reduction.sigrej import sigrej
from eureka.S4cal_StellarSpectra.s4cal_meta import S4cal_MetaClass

'''
import s4cal_calStellarSpec as s4cal
eventlabel = 'ngts10b_pc'
meta, spec, ds = s4cal.medianCalSpec(eventlabel)
'''

colors = ['xkcd:bright blue','purple','xkcd:soft green','orange']

def medianCalSpec(eventlabel, ecf_path=None, s3_meta=None, input_meta=None):
    '''Generate median calibrated stellar spectra using in-eclipse data
    and out-of-eclipse baseline.  The outputs also include the standard
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

    t_start = time_pkg.time()

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

    # Flag outliers in time
    mask = sigrej(spec.optspec.values, meta.sigma_thresh,
                  mask=~spec.optmask.values, axis=0)
    optspec = np.ma.masked_array(spec.optspec.values, ~mask)

    if isinstance(meta.t0, float):
        meta.t0 = [meta.t0]
    num_ecl = len(meta.t0)
    fig = None
    ax = None
    batch = []
    for i in range(num_ecl):
        # Compute baseline/in-transit median spectra
        t0 = meta.t0[i] + meta.time_offset
        p = meta.period
        rprs = meta.rprs
        ars = meta.ars
        cosi = np.cos(meta.inc*np.pi/180)
        # total transit duration
        if meta.t14 is None:
            meta.t14 = p/np.pi*np.arcsin(
                1/ars*np.sqrt(((1 + rprs)**2 - (ars*cosi)**2)/(1 - cosi**2)))
            if ~np.isfinite(meta.t14):
                raise Exception("t14 is not finite. Check your system " +
                                "parameters.")
        # Full transit duration
        if meta.t23 is None:
            meta.t23 = p/np.pi*np.arcsin(
                1/ars*np.sqrt(((1 - rprs)**2 - (ars*cosi)**2)/(1 - cosi**2)))
            if ~np.isfinite(meta.t23):
                raise Exception("t23 is not finite. Check your system " +
                                "parameters or planet may be grazing.")
        # Indeces for first through fourth contact
        it1 = np.where(spec.time > (t0 - meta.t14/2))[0][0]
        it2 = np.where(spec.time > (t0 - meta.t23/2))[0][0]
        it3 = np.where(spec.time > (t0 + meta.t23/2))[0][0]
        it4 = np.where(spec.time > (t0 + meta.t14/2))[0][0]
        # Indeces for beginning and end of baseline
        if meta.base_dur == None:
            it0 = 0
            it5 = -1
        else:
            try:
                it0 = np.where(spec.time > (t0 - meta.t14/2 - meta.base_dur))[0][0]
            except:
                it0 = 0
            try:
                it5 = np.where(spec.time > (t0 + meta.t14/2 + meta.base_dur))[0][0]
            except:
                it5 = -1
        meta.it = [it0, it1, it2, it3, it4, it5]

        if meta.isplots_S4cal >= 2:
            fig, ax = plot_whitelc(optspec, spec.time, meta, i, fig=fig, ax=ax)

        # Median baseline (i.e. out-of-eclipse) flux
        spec_baseline = np.ma.median(np.ma.concatenate(
            (optspec[it0:it1], optspec[it4:it5])), axis=0)
        std_baseline = np.ma.std(np.ma.concatenate(
            (optspec[it0:it1], optspec[it4:it5])), axis=0)
        # Mask outliers along wavelength axis
        tck = spi.splrep(wave, spec_baseline, w=1/std_baseline, k=3)
        spline = spi.splev(wave, tck)
        mask = sigrej(spec_baseline - spline, meta.sigma_thresh)
        igood = np.where(mask)[0]
        # Create XArray data arrays
        base_flux = xrio.makeLCDA(spec_baseline[igood, np.newaxis],
                                  wave[igood], [t0],
                                  spec.optspec.flux_units,
                                  spec.optspec.wave_units,
                                  spec.optspec.time_units,
                                  name='Median Baseline Flux')
        base_fstd = xrio.makeLCDA(std_baseline[igood, np.newaxis],
                                  wave[igood], [t0],
                                  spec.optspec.flux_units,
                                  spec.optspec.wave_units,
                                  spec.optspec.time_units,
                                  name='Std. Dev. of Baseline Flux')

        # Median in-eclipse (or in-transit) flux
        spec_t23 = np.ma.median(optspec[it2:it3+1], axis=0)
        std_t23 = np.ma.std(optspec[it2:it3+1], axis=0)
        # Mask outliers along wavelength axis
        tck = spi.splrep(wave, spec_t23, w=1/std_t23, k=3)
        spline = spi.splev(wave, tck)
        mask = sigrej(spec_t23 - spline, meta.sigma_thresh)
        igood = np.where(mask)[0]
        # Create XArray data arrays
        ecl_flux = xrio.makeLCDA(spec_t23[igood, np.newaxis],
                                 wave[igood], [t0],
                                 spec.optspec.flux_units,
                                 spec.optspec.wave_units,
                                 spec.optspec.time_units,
                                 name='Median In-Eclipse Flux')
        ecl_fstd = xrio.makeLCDA(std_t23[igood, np.newaxis],
                                 wave[igood], [t0],
                                 spec.optspec.flux_units,
                                 spec.optspec.wave_units,
                                 spec.optspec.time_units,
                                 name='Std. Dev. of In-Eclipse Flux')

        # Create XArray dataset
        ds = dict(base_flux=base_flux, base_fstd=base_fstd,
                  ecl_flux=ecl_flux, ecl_fstd=ecl_fstd)
        ds = xrio.makeDataset(ds)
        batch.append(ds)

    # Concatenate datasets
    ds = xrio.concat(batch, dim="time")
    filename = meta.outputdir+'S4cal_'+meta.eventlabel + "_CalStellarSpec.h5"
    xrio.writeXR(filename, ds, verbose=True)

    if meta.isplots_S4cal >= 2:
        plot_stellarSpec(meta, ds)

    return meta, spec, ds


def plot_whitelc(optspec, time, meta, i, fig=None, ax=None):
    '''Plot binned white light curve and indicate
    baseline and in-eclipse regions.
    '''
    toffset = meta.time_offset
    it0, it1, it2, it3, it4, it5 = meta.it

    # Created binned white LC
    lc = np.ma.sum(optspec, axis=1)
    lc_bin = util.binData_time(lc, time, nbin=meta.nbin_plot)
    time_bin = util.binData_time(time, time, nbin=meta.nbin_plot)

    if i == 0:
        fig = plt.figure(7202)
        plt.clf()
        ax = fig.subplots(1, 1)
        ax.plot(time_bin-toffset, lc_bin, '.', color='0.2', alpha=0.8,
                label='Binned White LC')
    ymin, ymax = ax.get_ylim()
    ax.vlines([time[it1]-toffset, time[it4]-toffset,
              time[it0]-toffset, time[it5]-toffset],
              ymin, ymax, color=colors[1], label='Baseline Regions')
    ax.vlines([time[it2]-toffset, time[it3]-toffset],
              ymin, ymax, color=colors[0], label='In-Eclipse Region')
    if i == 0:
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='best')
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Normalized Flux")
    fname = 'figs'+os.sep+'fig7202_WhiteLC'
    fig.savefig(meta.outputdir+fname+plots.figure_filetype,
                bbox_inches='tight', dpi=300)
    return fig, ax


def plot_stellarSpec(meta, ds):
    '''Plot calibrated stellar spectra from
    baseline and in-eclipse regions.
    '''
    fig = plt.figure(4201)
    plt.clf()
    ax = fig.subplots(1, 1)
    for i in range(len(ds.time)):
        ax.errorbar(ds.wavelength, ds.base_flux[:,i], ds.base_fstd[:,i],
                    fmt='.', ms=1, label=f'Baseline ({ds.time.values[i]})')
        ax.errorbar(ds.wavelength, ds.ecl_flux[:,i], ds.ecl_fstd[:,i],
                    fmt='.', ms=1, label=f'In-Eclipse ({ds.time.values[i]})')

    ax.legend(loc='best')
    ax.set_xlabel("Wavelength ($\mu$m)")
    ax.set_ylabel(f"Flux ({ds.base_flux.flux_units})")

    fname = 'figs'+os.sep+'fig4201_CalStellarSpec'
    fig.savefig(meta.outputdir+fname+plots.figure_filetype,
                bbox_inches='tight', dpi=300)
    return
