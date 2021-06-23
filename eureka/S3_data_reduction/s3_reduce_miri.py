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
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from . import optspex
from importlib import reload
from ..lib import astropytable
from ..lib import util
from . import plots_s3
import matplotlib.pyplot as plt
reload(optspex)


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



def image(data, meta, n):

    intstart, subdata, submask = data.intstart, data.subdata, data.submask

    plt.figure(3301)
    plt.clf()
    plt.suptitle(str(intstart + n))

    max = np.max(subdata[n] * submask[n])
    plt.imshow(subdata[n] * submask[n], origin='lower', aspect='auto', vmin=0, vmax=max / 10)

    plt.savefig(meta.workdir + '/figs/fig3301-' + str(intstart + n) + '-Image.png')




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
        #meta.src_xpos = data.shdr['SRCXPOS'] - meta.xwindow[0]
        #meta.src_ypos = data.shdr['SRCYPOS'] - meta.ywindow[0]
        # Record integration mid-times in BJD_TDB
        data.bjdtdb = data.int_times['int_mid_BJD_TDB']
        # Trim data to subarray region of interest
        data, meta = util.trim(data, meta)
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        data.submask = np.ones(data.subdata.shape)

        # Convert units (eg. for NIRCam: MJy/sr -> DN -> Electrons)
        #data, meta = inst.unit_convert(data, meta, log)

        # Check if arrays have NaNs
        data.submask = util.check_nans(data.subdata, data.submask, log)
        data.submask = util.check_nans(data.suberr, data.submask, log)
        data.submask = util.check_nans(data.subv0, data.submask, log)

        if meta.isplots_S3 >= 3:
            for n in range(meta.n_int):
                # make image+background plots
                image(data, meta, n)