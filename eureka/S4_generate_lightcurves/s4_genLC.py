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
from ..lib import savetable

def lcJWST(eventlabel, workdir, md=None, isplots=False):
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
    if md == None:
        md = me.load(workdir+'/S3_'+eventlabel+'_Meta_Save.dat')

    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(md, ecf)

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    md.lcdir = md.workdir + '/S4_' + datetime + '_' + str(md.nspecchan) + 'chan'
    if not os.path.exists(md.lcdir):
        os.makedirs(md.lcdir)
    if not os.path.exists(md.lcdir+"/figs"):
        os.makedirs(md.lcdir+"/figs")

    # Copy existing S3 log file
    md.s4_logname  = './'+md.lcdir + '/S4_' + md.eventlabel + ".log"
    #shutil.copyfile(ev.logname, ev.s4_logname, follow_symlinks=True)
    log         = logedit.Logedit(md.s4_logname, read=md.logname)
    log.writelog("\nStarting Stage 4: Generate Light Curves\n")

    table = savetable.readtable(md)

    optspec, wave_1d, bjdtdb = np.reshape(table['optspec'].data, (-1, md.subnx)), \
                               table['wave_1d'].data[0:md.subnx], table['bjdtdb'].data[::md.subnx]

    #Replace NaNs with zero
    optspec[np.where(np.isnan(optspec))] = 0



    # Determine wavelength bins
    binsize     = (md.wave_max - md.wave_min)/md.nspecchan
    md.wave_low = np.round([i for i in np.linspace(md.wave_min, md.wave_max-binsize, md.nspecchan)],3)
    md.wave_hi  = np.round([i for i in np.linspace(md.wave_min+binsize, md.wave_max, md.nspecchan)],3)

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
    md.lcdata   = np.zeros((md.nspecchan, n_int))
    md.lcerr    = np.zeros((md.nspecchan, n_int))
    # ev.eventname2 = ev.eventname
    for i in range(md.nspecchan):
        log.writelog(f"Bandpass {i} = %.3f - %.3f" % (md.wave_low[i],md.wave_hi[i]))
        # Compute valid indeces within wavelength range
        index   = np.where((wave_1d >= md.wave_low[i])*(wave_1d <= md.wave_hi[i]))[0]
        # Sum flux for each spectroscopic channel
        md.lcdata[i]    = np.sum(optspec[:,index],axis=1)
        # Add uncertainties in quadrature
        md.lcerr[i]     = np.sqrt(np.sum(optspec[:,index]**2,axis=1))

        # Plot each spectroscopic light curve
        if isplots >= 3:
            plt.figure(4100+i, figsize=(8,6))
            plt.clf()
            plt.suptitle(f"Bandpass {i}: %.3f - %.3f" % (md.wave_low[i],md.wave_hi[i]))
            ax = plt.subplot(111)
            mjd     = np.floor(bjdtdb[0])
            # Normalized light curve
            norm_lcdata = md.lcdata[i]/md.lcdata[i,-1]
            norm_lcerr  = md.lcerr[i]/md.lcdata[i,-1]
            plt.errorbar(bjdtdb-mjd, norm_lcdata, norm_lcerr, fmt='o', color=f'C{i}', mec='w')
            plt.text(0.05, 0.1, "MAD = "+str(np.round(1e6*np.median(np.abs(np.ediff1d(norm_lcdata)))))+" ppm", transform=ax.transAxes, color='k')
            plt.ylabel('Normalized Flux')
            plt.xlabel(f'Time [MJD + {mjd}]')

            plt.subplots_adjust(left=0.10,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.3)
            plt.savefig(md.lcdir + '/figs/Fig' + str(4100+i) + '-' + md.eventlabel + '-1D_LC.png')

    # Save results
    log.writelog('Saving results')
    me.saveevent(md, md.lcdir + '/S4_' + md.eventlabel + "_Meta_Save", save=[])

    # if (isplots >= 1) and (correctDrift == True):
    #     # Plot Drift
    #     # Plots corrected 2D light curves


    log.closelog()
    return md
