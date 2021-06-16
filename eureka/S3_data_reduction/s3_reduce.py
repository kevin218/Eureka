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
from . import optspex
from importlib import reload
from ..lib import astropytable
from ..lib import util
from . import plots_s3
reload(optspex)


class Metadata():
  def __init__(self):

    # initialize Univ
    #Univ.__init__(self)
    #self.initpars(ecf)
    #self.foo = 2
    return

class Data():
  def __init__(self):

    # initialize Univ
    #Univ.__init__(self)
    #self.initpars(ecf)
    #self.foo = 2
    return

def reduceJWST(eventlabel):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset

    Returns
    -------
    md          : Event object
    dat         : Data object

    Remarks
    -------


    History
    -------
    Written by Kevin Stevenson      May 2021

    '''

    t0      = time.time()

    # Initialize metadata object
    md              = Metadata()
    md.eventlabel   = eventlabel

    # Initialize data object
    dat              = Data()


    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    md.workdir = 'S3_' + datetime + '_' + md.eventlabel
    if not os.path.exists(md.workdir):
        os.makedirs(md.workdir)
    if not os.path.exists(md.workdir+"/figs"):
        os.makedirs(md.workdir+"/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(md, ecf)

    # Load instrument module
    exec('from . import ' + md.inst + ' as inst', globals())
    reload(inst)

    # Open new log file
    md.logname  = './'+md.workdir + '/S3_' + md.eventlabel + ".log"
    log         = logedit.Logedit(md.logname)
    log.writelog("\nStarting Stage 3 Reduction")

    # Create list of file segments
    md = util.readfiles(md)
    num_data_files = len(md.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {md.suffix}.fits')

    stdspec = np.array([])
    # Loop over each segment
    if md.testing_S3 == True:
        istart = num_data_files-1
    else:
        istart = 0
    for m in range(istart, num_data_files):
        # Report progress

        # Read in data frame and header
        log.writelog(f'Reading file {m+1} of {num_data_files}')
        dat = inst.read(md.segment_list[m], dat, returnHdr=True)
        # Get number of integrations and frame dimensions
        md.n_int, md.ny, md.nx = dat.data.shape
        # Locate source postion
        md.src_xpos = dat.shdr['SRCXPOS']-md.xwindow[0]
        md.src_ypos = dat.shdr['SRCYPOS']-md.ywindow[0]
        # Record integration mid-times in BJD_TDB
        dat.bjdtdb = dat.int_times['int_mid_BJD_TDB']
        # Trim data to subarray region of interest
        dat, md = util.trim(dat, md)
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        dat.submask = np.ones(dat.subdata.shape)

        #Convert units (eg. for NIRCam: MJy/sr -> DN -> Electrons)
        dat, md = inst.unit_convert(dat, md, log)

        # Check if arrays have NaNs
        dat.submask = util.check_nans(dat.subdata, dat.submask, log)
        dat.submask = util.check_nans(dat.suberr, dat.submask, log)
        dat.submask = util.check_nans(dat.subv0, dat.submask, log)

        # Manually mask regions [colstart, colend, rowstart, rowend]
        if hasattr(md, 'manmask'):
            log.writelog("  Masking manually identified bad pixels")
            for i in range(len(md.manmask)):
                ind, colstart, colend, rowstart, rowend = md.manmask[i]
                dat.submask[rowstart:rowend,colstart:colend] = 0

        # Perform outlier rejection of sky background along time axis
        log.writelog('Performing background outlier rejection')
        md.bg_y1    = int(md.src_ypos - md.bg_hw)
        md.bg_y2    = int(md.src_ypos + md.bg_hw)
        dat.submask = inst.flag_bg(dat, md)


        dat = util.BGsubtraction(dat, md, log, md.isplots_S3)


        # Calulate drift2D
        # print("Calculating 2D drift...")

        # print("Performing rough, pixel-scale drift correction...")

        # Outlier rejection of full frame along time axis
        # print("Performing full-frame outlier rejection...")

        if md.isplots_S3 >= 3:
            for n in range(md.n_int):
                #make image+background plots
                plots_s3.image_and_background(dat, md, n)


        # print("Performing sub-pixel drift correction...")

        # Select only aperture region
        ap_y1       = int(md.src_ypos - md.spec_hw)
        ap_y2       = int(md.src_ypos + md.spec_hw)
        dat.apdata      = dat.subdata[:,ap_y1:ap_y2]
        dat.aperr       = dat.suberr [:,ap_y1:ap_y2]
        dat.apmask      = dat.submask[:,ap_y1:ap_y2]
        dat.apbg        = dat.subbg  [:,ap_y1:ap_y2]
        dat.apv0        = dat.subv0  [:,ap_y1:ap_y2]
        # Extract standard spectrum and its variance
        dat.stdspec     = np.sum(dat.apdata, axis=1)
        dat.stdvar      = np.sum(dat.aperr**2, axis=1)  #FINDME: stdvar >> stdspec, which is a problem
        # Compute fraction of masked pixels within regular spectral extraction window
        #numpixels   = 2.*md.spec_width*subnx
        #fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

        # Compute median frame
        md.medsubdata   = np.median(dat.subdata, axis=0)
        md.medapdata    = np.median(dat.apdata, axis=0)

        # Extract optimal spectrum with uncertainties
        log.writelog("  Performing optimal spectral extraction")
        dat.optspec     = np.zeros((dat.stdspec.shape))
        dat.opterr      = np.zeros((dat.stdspec.shape))
        gain        = 1         #FINDME: need to determine correct gain
        for n in range(md.n_int):
            dat.optspec[n], dat.opterr[n], mask = optspex.optimize(dat.apdata[n], dat.apmask[n], dat.apbg[n], dat.stdspec[n], gain, dat.apv0[n], p5thresh=md.p5thresh, p7thresh=md.p7thresh, fittype=md.fittype, window_len=md.window_len, deg=md.prof_deg, n=dat.intstart+n, isplots=md.isplots_S3, eventdir=md.workdir, meddata=md.medapdata)

        # Plotting results
        if md.isplots_S3 >= 3:
            for n in range(md.n_int):
                #make optimal spectrum plot
                plots_s3.optimal_spectrum(dat, md, n)

        # Append results
        if len(stdspec) == 0:
            wave_2d  = dat.subwave
            wave_1d  = dat.subwave[md.src_ypos]
            stdspec  = dat.stdspec
            stdvar   = dat.stdvar
            optspec  = dat.optspec
            opterr   = dat.opterr
            bjdtdb   = dat.bjdtdb
        else:
            stdspec  = np.append(stdspec, dat.stdspec, axis=0)
            stdvar   = np.append(stdvar, dat.stdvar, axis=0)
            optspec  = np.append(optspec, dat.optspec, axis=0)
            opterr   = np.append(opterr, dat.opterr, axis=0)
            bjdtdb   = np.append(bjdtdb, dat.bjdtdb, axis=0)

    # Calculate total time
    total = (time.time() - t0)/60.
    log.writelog('\nTotal time (min): ' + str(np.round(total,2)))


    # Save results
    log.writelog('Saving results')
    me.saveevent(md, md.workdir + '/S3_' + md.eventlabel + "_Meta_Save", save=[])

    # Save results
    log.writelog('Saving results')
    me.saveevent(dat, md.workdir + '/S3_' + md.eventlabel + "_Data_Save", save=[])

    log.writelog('Saving results as astropy table...')
    astropytable.savetable(md, bjdtdb, wave_1d, stdspec, stdvar, optspec, opterr)

    log.writelog('Generating figures')
    if md.isplots_S3 >= 1:
        # 2D light curve without drift correction
        plots_s3.lc_nodriftcorr(md, wave_1d, optspec)

    log.closelog()
    return md
