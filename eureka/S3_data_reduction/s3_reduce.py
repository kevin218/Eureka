#! /usr/bin/env python

# Generic Stage 3 reduction pipeline

"""
# Proposed Steps
# -------- -----
# 1.  Read in all data frames and header info from Stage 2 data products DONE
# 2.  Record JD and other relevant header information DONE
# 3.  Apply light-time correction (if necessary) DONE
# 4.  Calculate trace and 1D+2D wavelength solutions (if necessary)
# 5.  Make flats, apply flat field correction (Stage 2)
# 6.  Manually mask regions DONE
# 7.  Compute difference frames OR slopes (Stage 1)
# 8.  Perform outlier rejection of BG region DONE
# 9.  Background subtraction DONE
# 10. Compute 2D drift, apply rough (integer-pixel) correction
# 11. Full-frame outlier rejection for time-series stack of NDRs
# 12. Apply sub-pixel 2D drift correction
# 13. Extract spectrum through summation DONE
# 14. Compute median frame DONE
# 15. Optimal spectral extraction DONE
# 16. Save Stage 3 data products
# 17. Produce plots DONE
"""

import sys, os, time
import numpy as np
import multiprocessing as mp
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from . import bright2flux as b2f
from . import optspex
from importlib import reload
from ..lib import savetable
from ..lib import util
from . import plots_s3
reload(optspex)
reload(b2f)

class Event():
  def __init__(self):

    # initialize Univ
    #Univ.__init__(self)
    #self.initpars(ecf)
    #self.foo = 2
    return

def reduceJWST(eventlabel, isplots=False, testing=False):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset
    isplots     : boolean, Set True to produce plots

    Returns
    -------
    ev          : Event object

    Remarks
    -------


    History
    -------
    Written by Kevin Stevenson      May 2021

    '''

    t0      = time.time()

    # Initialize event object
    ev              = Event()
    ev.eventlabel   = eventlabel

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    ev.workdir = 'S3_' + datetime + '_' + ev.eventlabel
    if not os.path.exists(ev.workdir):
        os.makedirs(ev.workdir)
    if not os.path.exists(ev.workdir+"/figs"):
        os.makedirs(ev.workdir+"/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(ev, ecf)

    # Load instrument module
    exec('from . import ' + ev.inst + ' as inst', globals())
    reload(inst)

    # Open new log file
    ev.logname  = './'+ev.workdir + '/S3_' + ev.eventlabel + ".log"
    log         = logedit.Logedit(ev.logname)
    log.writelog("\nStarting Stage 3 Reduction")

    # Create list of file segments
    ev = util.readfiles(ev)
    num_data_files = len(ev.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {ev.suffix}.fits')

    ev.stdspec = np.array([])
    # Loop over each segment
    if testing == True:
        istart = num_data_files-1
    else:
        istart = 0
    for m in range(istart, num_data_files):
        # Report progress

        # Read in data frame and header
        log.writelog(f'Reading file {m+1} of {num_data_files}')
        data, err, dq, wave, v0, int_times, mhdr, shdr = inst.read(ev.segment_list[m], returnHdr=True)
        # Get number of integrations and frame dimensions
        n_int, ny, nx = data.shape
        intstart = mhdr['INTSTART']
        # Locate source postion
        src_xpos = shdr['SRCXPOS']-ev.xwindow[0]
        src_ypos = shdr['SRCYPOS']-ev.ywindow[0]
        # Record integration mid-times in BJD_TDB
        bjdtdb = int_times['int_mid_BJD_TDB']
        # Trim data to subarray region of interest
        subdata, suberr, subdq, subwave, subv0, subny, subnx = util.trim(ev, data,err, dq, wave, v0)
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        submask = np.ones(subdata.shape)
        if shdr['BUNIT'] == 'MJy/sr':
            # Convert from brightness units (MJy/sr) to flux units (uJy/pix)
            #log.writelog('Converting from brightness to flux units')
            #subdata, suberr, subv0 = b2f.bright2flux(subdata, suberr, subv0, shdr['PIXAR_A2'])
            # Convert from brightness units (MJy/sr) to DNs
            log.writelog('  Converting from brightness units (MJy/sr) to electrons')
            photfile = ev.topdir + ev.ancildir +'/'+ mhdr['R_PHOTOM'][7:]
            subdata, suberr, subv0 = b2f.bright2dn(subdata, suberr, subv0, subwave, photfile, mhdr, shdr)
            gainfile = ev.topdir + ev.ancildir +'/'+ mhdr['R_GAIN'][7:]
            subdata, suberr, subv0 = b2f.dn2electrons(subdata, suberr, subv0, gainfile, mhdr, ev.ywindow, ev.xwindow)

        # Check if arrays have NaNs
        submask = util.check_nans(subdata, submask, log)
        submask = util.check_nans(suberr, submask, log)
        submask = util.check_nans(subv0, submask, log)

        # Manually mask regions [colstart, colend, rowstart, rowend]
        if hasattr(ev, 'manmask'):
            log.writelog("  Masking manually identified bad pixels")
            for i in range(len(ev.manmask)):
                ind, colstart, colend, rowstart, rowend = ev.manmask[i]
                submask[rowstart:rowend,colstart:colend] = 0

        # Perform outlier rejection of sky background along time axis
        log.writelog('  Performing background outlier rejection')
        bg_y1    = int(src_ypos - ev.bg_hw)
        bg_y2    = int(src_ypos + ev.bg_hw)
        submask = inst.flag_bg(subdata, suberr, submask, bg_y1, bg_y2, ev.bg_thresh)


        subbg, submask, subdata = util.BGsubtraction(ev, log, n_int, bg_y1, bg_y2,subdata, submask, isplots)


        # Calulate drift2D
        # print("Calculating 2D drift...")

        # print("Performing rough, pixel-scale drift correction...")

        # Outlier rejection of full frame along time axis
        # print("Performing full-frame outlier rejection...")

        if isplots >= 3:
            for n in range(n_int):
                #make image+background plots
                plots_s3.image_and_background(ev, intstart, n, subdata, submask, subbg)


        # print("Performing sub-pixel drift correction...")

        # Select only aperture region
        ap_y1       = int(src_ypos - ev.spec_hw)
        ap_y2       = int(src_ypos + ev.spec_hw)
        apdata      = subdata[:,ap_y1:ap_y2]
        aperr       = suberr [:,ap_y1:ap_y2]
        apmask      = submask[:,ap_y1:ap_y2]
        apbg        = subbg  [:,ap_y1:ap_y2]
        apv0        = subv0  [:,ap_y1:ap_y2]
        # Extract standard spectrum and its variance
        stdspec     = np.sum(apdata, axis=1)
        stdvar      = np.sum(aperr**2, axis=1)  #FINDME: stdvar >> stdspec, which is a problem
        # Compute fraction of masked pixels within regular spectral extraction window
        #numpixels   = 2.*ev.spec_width*subnx
        #fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

        # Compute median frame
        ev.medsubdata   = np.median(subdata, axis=0)
        ev.medapdata    = np.median(apdata, axis=0)

        # Extract optimal spectrum with uncertainties
        log.writelog("  Performing optimal spectral extraction")
        optspec     = np.zeros((stdspec.shape))
        opterr      = np.zeros((stdspec.shape))
        gain        = 1         #FINDME: need to determine correct gain
        for n in range(n_int):
            optspec[n], opterr[n], mask = optspex.optimize(apdata[n], apmask[n], apbg[n], stdspec[n], gain, apv0[n], p5thresh=ev.p5thresh, p7thresh=ev.p7thresh, fittype=ev.fittype, window_len=ev.window_len, deg=ev.prof_deg, n=intstart+n, isplots=isplots, eventdir=ev.workdir, meddata=ev.medapdata)

        # Plotting results
        if isplots >= 3:
            for n in range(n_int):
                #make optimal spectrum plot
                plots_s3.optimal_spectrum(ev, intstart, n, subnx, stdspec, optspec, opterr)

        # Append results
        if len(ev.stdspec) == 0:
            ev.wave_2d  = subwave
            ev.wave_1d  = subwave[src_ypos]
            ev.stdspec  = stdspec
            ev.stdvar   = stdvar
            ev.optspec  = optspec
            ev.opterr   = opterr
            ev.bjdtdb   = bjdtdb
        else:
            ev.stdspec  = np.append(ev.stdspec, stdspec, axis=0)
            ev.stdvar   = np.append(ev.stdvar, stdvar, axis=0)
            ev.optspec  = np.append(ev.optspec, optspec, axis=0)
            ev.opterr   = np.append(ev.opterr, opterr, axis=0)
            ev.bjdtdb   = np.append(ev.bjdtdb, bjdtdb, axis=0)

    # Calculate total time
    total = (time.time() - t0)/60.
    log.writelog('\nTotal time (min): ' + str(np.round(total,2)))


    # Save results
    log.writelog('Saving results')
    me.saveevent(ev, ev.workdir + '/S3_' + ev.eventlabel + "_Save", save=[])

    #log.writelog('Saving results as astropy table...')
    #savetable.savetable(ev)

    log.writelog('Generating figures')
    if isplots >= 1:
        # 2D light curve without drift correction
        plots_s3.lc_nodriftcorr(ev)

    log.closelog()
    return ev
