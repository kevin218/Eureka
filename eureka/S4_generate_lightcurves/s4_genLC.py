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

def lcJWST(eventlabel, workdir, ev=None, isplots=False):
    #expand=1, smooth_len=None, correctDrift=True
    '''
    Compute photometric flux over specified range of wavelengths

    Parameters
    ----------
    eventlabel  : Unique identifier for these data
    workdir     : Location of save file
    ev          : event object
    isplots     : Set True to produce plots

    Returns
    -------
    event object

    History
    -------
    Written by Kevin Stevenson      June 2021

    '''
    #load savefile
    if ev == None:
        ev = me.load(workdir+'/S3_'+eventlabel+'_Save.dat')

    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(ev, ecf)

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    ev.lcdir = ev.workdir + '/S4_' + datetime + '_' + str(ev.nspecchan) + 'chan'
    if not os.path.exists(ev.lcdir):
        os.makedirs(ev.lcdir)
    if not os.path.exists(ev.lcdir+"/figs"):
        os.makedirs(ev.lcdir+"/figs")

    # Copy existing S3 log file
    ev.s4_logname  = './'+ev.lcdir + '/S4_' + ev.eventlabel + ".log"
    #shutil.copyfile(ev.logname, ev.s4_logname, follow_symlinks=True)
    log         = logedit.Logedit(ev.s4_logname, read=ev.logname)
    log.writelog("\nStarting Stage 4: Generate Light Curves\n")

    #Replace NaNs with zero
    ev.optspec[np.where(np.isnan(ev.optspec))] = 0

    # Determine wavelength bins
    binsize     = (ev.wave_max - ev.wave_min)/ev.nspecchan
    ev.wave_low = np.round([i for i in np.linspace(ev.wave_min, ev.wave_max-binsize, ev.nspecchan)],3)
    ev.wave_hi  = np.round([i for i in np.linspace(ev.wave_min+binsize, ev.wave_max, ev.nspecchan)],3)

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
    n_int, nx   = ev.optspec.shape
    ev.lcdata   = np.zeros((ev.nspecchan, n_int))
    ev.lcerr    = np.zeros((ev.nspecchan, n_int))
    # ev.eventname2 = ev.eventname
    for i in range(ev.nspecchan):
        log.writelog(f"Bandpass {i} = %.3f - %.3f" % (ev.wave_low[i],ev.wave_hi[i]))
        # Compute valid indeces within wavelength range
        index   = np.where((ev.wave_1d >= ev.wave_low[i])*(ev.wave_1d <= ev.wave_hi[i]))[0]
        # Sum flux for each spectroscopic channel
        ev.lcdata[i]    = np.sum(ev.optspec[:,index],axis=1)
        # Add uncertainties in quadrature
        ev.lcerr[i]     = np.sqrt(np.sum(ev.optspec[:,index]**2,axis=1))

        # Plot each spectroscopic light curve
        if isplots >= 3:
            plt.figure(4100+i, figsize=(8,6))
            plt.clf()
            plt.suptitle(f"Bandpass {i}: %.3f - %.3f" % (ev.wave_low[i],ev.wave_hi[i]))
            ax = plt.subplot(111)
            mjd     = np.floor(ev.bjdtdb[0])
            # Normalized light curve
            norm_lcdata = ev.lcdata[i]/ev.lcdata[i,-1]
            norm_lcerr  = ev.lcerr[i]/ev.lcdata[i,-1]
            plt.errorbar(ev.bjdtdb-mjd, norm_lcdata, norm_lcerr, fmt='o', color=f'C{i}', mec='w')
            plt.text(0.05, 0.1, "MAD = "+str(np.round(1e6*np.median(np.abs(np.ediff1d(norm_lcdata)))))+" ppm", transform=ax.transAxes, color='k')
            plt.ylabel('Normalized Flux')
            plt.xlabel(f'Time [MJD + {mjd}]')

            plt.subplots_adjust(left=0.10,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.3)
            plt.savefig(ev.lcdir + '/figs/Fig' + str(4100+i) + '-' + ev.eventlabel + '-1D_LC.png')

    # Save results
    log.writelog('Saving results')
    me.saveevent(ev, ev.lcdir + '/S4_' + ev.eventlabel + "_Save", save=[])

    # if (isplots >= 1) and (correctDrift == True):
    #     # Plot Drift
    #     # Plots corrected 2D light curves


    log.closelog()
    return ev
