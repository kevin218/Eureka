#! /usr/bin/env python

# Generic Stage 4 light curve generation pipeline

"""
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
"""
import sys, os, time, shutil
import numpy as np
import matplotlib.pyplot as plt
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from ..lib import astropytable
from . import plots_s4

def lcJWST(eventlabel, workdir, meta=None):
    #expand=1, smooth_len=None, correctDrift=True
    '''
    Compute photometric flux over specified range of wavelengths

    Parameters
    ----------
    eventlabel  : Unique identifier for these data
    workdir     : Location of save file
    meta          : metadata object

    Returns
    -------
    event object

    History
    -------
    Written by Kevin Stevenson      June 2021

    '''
    #load savefile
    if meta == None:
        meta = me.load(workdir + '/S3_' + eventlabel + '_Meta_Save.dat')

    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    meta.lcdir = meta.workdir + '/S4_' + datetime + '_' + str(meta.nspecchan) + 'chan'
    if not os.path.exists(meta.lcdir):
        os.makedirs(meta.lcdir)
    if not os.path.exists(meta.lcdir+"/figs"):
        os.makedirs(meta.lcdir+"/figs")

    # Copy existing S3 log file
    meta.s4_logname  = './' + meta.lcdir + '/S4_' + meta.eventlabel + ".log"
    #shutil.copyfile(ev.logname, ev.s4_logname, follow_symlinks=True)
    log         = logedit.Logedit(meta.s4_logname, read=meta.logname)
    log.writelog("\nStarting Stage 4: Generate Light Curves\n")

    table = astropytable.readtable(meta)

    # Reverse the reshaping which has been done when saving the astropy table
    optspec, wave_1d, bjdtdb = np.reshape(table['optspec'].data, (-1, meta.subnx)), \
                               table['wave_1d'].data[0:meta.subnx], table['bjdtdb'].data[::meta.subnx]

    #Replace NaNs with zero
    optspec[np.where(np.isnan(optspec))] = 0



    # Determine wavelength bins
    binsize     = (meta.wave_max - meta.wave_min)/meta.nspecchan
    meta.wave_low = np.round([i for i in np.linspace(meta.wave_min, meta.wave_max-binsize, meta.nspecchan)],3)
    meta.wave_hi  = np.round([i for i in np.linspace(meta.wave_min+binsize, meta.wave_max, meta.nspecchan)],3)

    # Apply 1D drift correction
    # if correctDrift == True:
    #     #Shift 1D spectra
    #     #Calculate drift over all frames and non-destructive reads
    #     log.writelog('Applying drift correction...')
    #     ev.drift, ev.goodmask = hst.drift_fit2(ev, preclip=0, postclip=None, width=5*expand, deg=2, validRange=11*expand, istart=istart, iref=ev.iref[0])
    #     # Correct for drift
    #     for m in range(ev.n_files):
    #         for n in range(istart,ev.n_reads-1):
    #             spline            = spi.UnivariateSpline(np.arange(nx), ev.spectra[m,n], k=3, s=0)
    #             ev.spectra[m,n] = spline(np.arange(nx)+ev.drift[m,n])

    log.writelog("Generating light curves")
    n_int, nx   = optspec.shape
    meta.lcdata   = np.zeros((meta.nspecchan, n_int))
    meta.lcerr    = np.zeros((meta.nspecchan, n_int))
    # ev.eventname2 = ev.eventname
    for i in range(meta.nspecchan):
        log.writelog(f"Bandpass {i} = %.3f - %.3f" % (meta.wave_low[i], meta.wave_hi[i]))
        # Compute valid indeces within wavelength range
        index   = np.where((wave_1d >= meta.wave_low[i])*(wave_1d <= meta.wave_hi[i]))[0]
        # Sum flux for each spectroscopic channel
        meta.lcdata[i]    = np.sum(optspec[:,index],axis=1)
        # Add uncertainties in quadrature
        meta.lcerr[i]     = np.sqrt(np.sum(optspec[:,index]**2,axis=1))

        # Plot each spectroscopic light curve
        if meta.isplots_S4 >= 3:
            plots_s4.binned_lightcurve(meta, bjdtdb, i)

    # Save results
    log.writelog('Saving results')
    me.saveevent(meta, meta.lcdir + '/S4_' + meta.eventlabel + "_Meta_Save", save=[])

    # if (isplots >= 1) and (correctDrift == True):
    #     # Plot Drift
    #     # Plots corrected 2D light curves

    # Copy ecf
    log.writelog('Copy S4 ecf')
    shutil.copy(ecffile, meta.lcdir)

    log.closelog()
    return meta
