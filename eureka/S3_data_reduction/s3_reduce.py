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

import os, time
import numpy as np
import shutil
from . import optspex
from . import plots_s3
from . import background as bg
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from ..lib import astropytable
from ..lib import util
from importlib import reload
reload(optspex)
reload(bg)


class MetaClass:
    def __init__(self):
        # initialize Univ
        # Univ.__init__(self)
        # self.initpars(ecf)
        # self.foo = 2
        return


class DataClass:
    def __init__(self):
        # initialize Univ
        # Univ.__init__(self)
        # self.initpars(ecf)
        # self.foo = 2
        return


def reduceJWST(eventlabel):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset

    Returns
    -------
    meta          : Metadata object
    data         : Data object

    Remarks
    -------


    History
    -------
    Written by Kevin Stevenson      May 2021

    '''

    t0 = time.time()

    # Initialize metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Initialize data object
    data = DataClass()

    # Create directories for Stage 3 processing
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S')
    meta.workdir = 'S3_' + datetime + '_' + meta.eventlabel
    if not os.path.exists(meta.workdir):
        os.makedirs(meta.workdir)
    if not os.path.exists(meta.workdir + "/figs"):
        os.makedirs(meta.workdir + "/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Load instrument module
    exec('from . import ' + meta.inst + ' as inst', globals())
    reload(inst)

    # Open new log file
    meta.logname = './' + meta.workdir + '/S3_' + meta.eventlabel + ".log"
    log = logedit.Logedit(meta.logname)
    log.writelog("\nStarting Stage 3 Reduction")

    # Create list of file segments
    meta = util.readfiles(meta)
    num_data_files = len(meta.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {meta.suffix}.fits')

    stdspec = np.array([])
    # Loop over each segment
    # Only reduce the last segment/file if testing_S3 is set to True in ecf
    if meta.testing_S3:
        istart = num_data_files - 1
    else:
        istart = 0
    for m in range(istart, num_data_files):
        # Report progress

        # Read in data frame and header
        log.writelog(f'Reading file {m + 1} of {num_data_files}')
        data = inst.read(meta.segment_list[m], data)
        # Get number of integrations and frame dimensions
        meta.n_int, meta.ny, meta.nx = data.data.shape
        # Locate source postion
        meta.src_xpos = data.shdr['SRCXPOS'] - meta.xwindow[0]
        meta.src_ypos = data.shdr['SRCYPOS'] - meta.ywindow[0]
        # Record integration mid-times in BJD_TDB
        data.bjdtdb = data.int_times['int_mid_BJD_TDB']
        # Trim data to subarray region of interest
        data, meta = util.trim(data, meta)
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        data.submask = np.ones(data.subdata.shape)

        # Convert units (eg. for NIRCam: MJy/sr -> DN -> Electrons)
        data, meta = inst.unit_convert(data, meta, log)

        # Check if arrays have NaNs
        data.submask = util.check_nans(data.subdata, data.submask, log)
        data.submask = util.check_nans(data.suberr, data.submask, log)
        data.submask = util.check_nans(data.subv0, data.submask, log)

        # Manually mask regions [colstart, colend, rowstart, rowend]
        if hasattr(meta, 'manmask'):
            log.writelog("  Masking manually identified bad pixels")
            for i in range(len(meta.manmask)):
                ind, colstart, colend, rowstart, rowend = meta.manmask[i]
                data.submask[rowstart:rowend, colstart:colend] = 0

        # Perform outlier rejection of sky background along time axis
        log.writelog('  Performing background outlier rejection')
        meta.bg_y1 = int(meta.src_ypos - meta.bg_hw)
        meta.bg_y2 = int(meta.src_ypos + meta.bg_hw)
        data = inst.flag_bg(data, meta)

        data = bg.BGsubtraction(data, meta, log, meta.isplots_S3)

        # Calulate drift2D
        # print("Calculating 2D drift...")

        # print("Performing rough, pixel-scale drift correction...")

        # Outlier rejection of full frame along time axis
        # print("Performing full-frame outlier rejection...")

        if meta.isplots_S3 >= 3:
            for n in range(meta.n_int):
                # make image+background plots
                plots_s3.image_and_background(data, meta, n)

        # print("Performing sub-pixel drift correction...")

        # Select only aperture region
        ap_y1 = int(meta.src_ypos - meta.spec_hw)
        ap_y2 = int(meta.src_ypos + meta.spec_hw)
        data.apdata  = data.subdata[:, ap_y1:ap_y2]
        data.aperr   = data.suberr[:, ap_y1:ap_y2]
        data.apmask  = data.submask[:, ap_y1:ap_y2]
        data.apbg    = data.subbg[:, ap_y1:ap_y2]
        data.apv0    = data.subv0[:, ap_y1:ap_y2]
        # Extract standard spectrum and its variance
        data.stdspec = np.sum(data.apdata, axis=1)
        data.stdvar  = np.sum(data.aperr ** 2, axis=1)  # FINDME: stdvar >> stdspec, which is a problem
        # Compute fraction of masked pixels within regular spectral extraction window
        # numpixels   = 2.*meta.spec_width*subnx
        # fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

        # Compute median frame
        data.medsubdata = np.median(data.subdata, axis=0)
        data.medapdata  = np.median(data.apdata, axis=0)

        # Extract optimal spectrum with uncertainties
        log.writelog("  Performing optimal spectral extraction")
        data.optspec = np.zeros(data.stdspec.shape)
        data.opterr  = np.zeros(data.stdspec.shape)
        gain = 1  # FINDME: need to determine correct gain
        for n in range(meta.n_int):
            data.optspec[n], data.opterr[n], mask = optspex.optimize(data.apdata[n], data.apmask[n], data.apbg[n],
                                                                     data.stdspec[n], gain, data.apv0[n],
                                                                     p5thresh=meta.p5thresh, p7thresh=meta.p7thresh,
                                                                     fittype=meta.fittype, window_len=meta.window_len,
                                                                     deg=meta.prof_deg, n=data.intstart + n,
                                                                     isplots=meta.isplots_S3, eventdir=meta.workdir,
                                                                     meddata=data.medapdata)

        # Plotting results
        if meta.isplots_S3 >= 3:
            for n in range(meta.n_int):
                # make optimal spectrum plot
                plots_s3.optimal_spectrum(data, meta, n)

        # Append results
        if len(stdspec) == 0:
            wave_2d = data.subwave
            wave_1d = data.subwave[meta.src_ypos]
            stdspec = data.stdspec
            stdvar  = data.stdvar
            optspec = data.optspec
            opterr  = data.opterr
            bjdtdb  = data.bjdtdb
        else:
            stdspec = np.append(stdspec, data.stdspec, axis=0)
            stdvar  = np.append(stdvar, data.stdvar, axis=0)
            optspec = np.append(optspec, data.optspec, axis=0)
            opterr  = np.append(opterr, data.opterr, axis=0)
            bjdtdb  = np.append(bjdtdb, data.bjdtdb, axis=0)

    # Calculate total time
    total = (time.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # Save results
    log.writelog('Saving Metadata')
    me.saveevent(meta, meta.workdir + '/S3_' + meta.eventlabel + "_Meta_Save", save=[])

    # Save results
    #log.writelog('Saving results')
    #me.saveevent(data, meta.workdir + '/S3_' + meta.eventlabel + "_Data_Save", save=[])

    log.writelog('Saving results as astropy table')
    astropytable.savetable_S3(meta, bjdtdb, wave_1d, stdspec, stdvar, optspec, opterr)

    # Copy ecf
    log.writelog('Copy S3 ecf')
    shutil.copy(ecffile, meta.workdir)

    if meta.isplots_S3 >= 1:
        log.writelog('Generating figure')
        # 2D light curve without drift correction
        plots_s3.lc_nodriftcorr(meta, wave_1d, optspec)

    log.closelog()
    return meta
