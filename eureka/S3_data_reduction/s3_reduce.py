#! /usr/bin/env python

# Eureka! Stage 3 reduction pipeline

# Proposed Steps
# --------------
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


import os, glob
import time as time_pkg
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from . import optspex
from . import plots_s3, source_pos
from . import background as bg
from . import bright2flux as b2f
from ..lib import sort_nicely as sn
from ..lib import logedit
from ..lib import readECF
from ..lib import manageevent as me
from ..lib import astropytable
from ..lib import util

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return


class DataClass:
    '''A class to hold Eureka! image data.
    '''

    def __init__(self):
        return


def reduce(eventlabel, ecf_path='./', s2_meta=None):
    '''Reduces data images and calculates optimal spectra.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    ecf_path:   str
        The absolute or relative path to where ecfs are stored
    s2_meta:    MetaClass
        The metadata object from Eureka!'s S2 step (if running S2 and S3 sequentially).

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S3.

    Notes
    -------
    History:

    - May 2021 Kevin Stevenson
        Initial version
    - October 2021 Taylor Bell
        Updated to allow for inputs from S2
    '''
    # Initialize data object
    data = DataClass()

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel

    if s2_meta == None:
        #load savefile
        s2_meta = read_s2_meta(meta)

    if s2_meta != None:
        meta = load_general_s2_meta_info(meta, ecf_path, s2_meta)
    else:
        meta.inputdir_raw = meta.inputdir
        meta.outputdir_raw = meta.outputdir
        meta.inputdir = os.path.join(meta.topdir, *meta.inputdir_raw.split(os.sep))
        meta.outputdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))

    # check for range of spectral apertures
    if isinstance(meta.spec_hw, list):
        meta.spec_hw_range = range(meta.spec_hw[0], meta.spec_hw[1]+meta.spec_hw[2], meta.spec_hw[2])
    else:
        meta.spec_hw_range = [meta.spec_hw]

    #check for range of background apertures
    if isinstance(meta.bg_hw, list):
        meta.bg_hw_range = range(meta.bg_hw[0], meta.bg_hw[1]+meta.bg_hw[2], meta.bg_hw[2])
    else:
        meta.bg_hw_range = [meta.bg_hw]

    # create directories to store data
    meta.runs = [] # Used to make sure we're always looking at the right run for each aperture/annulus pair
    for spec_hw_val in meta.spec_hw_range:

        for bg_hw_val in meta.bg_hw_range:

            meta.eventlabel = eventlabel

            meta.runs.append(util.makedirectory(meta, 'S3', ap=spec_hw_val, bg=bg_hw_val))

    # begin process
    run_i = 0
    for spec_hw_val in meta.spec_hw_range:

        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            meta.outputdir = util.pathdirectory(meta, 'S3', meta.runs[run_i], ap=spec_hw_val, bg=bg_hw_val)
            run_i += 1

            event_ap_bg = meta.eventlabel + "_ap" + str(spec_hw_val) + '_bg' + str(bg_hw_val)

            # Open new log file
            meta.s3_logname = meta.outputdir + 'S3_' + event_ap_bg + ".log"
            if s2_meta != None:
                log = logedit.Logedit(meta.s3_logname, read=s2_meta.s2_logname)
            else:
                log = logedit.Logedit(meta.s3_logname)
            log.writelog("\nStarting Stage 3 Reduction\n")
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")
            log.writelog("Using ap=" + str(spec_hw_val) + ", bg=" + str(bg_hw_val))

            # Copy ecf
            log.writelog('Copying S3 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            # Create list of file segments
            meta = util.readfiles(meta)
            meta.num_data_files = len(meta.segment_list)
            if meta.num_data_files==0:
                log.writelog(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{meta.inputdir}"!', mute=True)
                raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{meta.inputdir}"!')
            else:
                log.writelog(f'\nFound {meta.num_data_files} data file(s) ending in {meta.suffix}.fits', mute=(not meta.verbose))

            with fits.open(meta.segment_list[-1]) as hdulist:
                # Figure out which instrument we are using
                meta.inst = hdulist[0].header['INSTRUME'].lower()
            # Load instrument module
            if meta.inst == 'miri':
                from . import miri as inst
            elif meta.inst == 'nircam':
                from . import nircam as inst
            elif meta.inst == 'nirspec':
                from . import nirspec as inst
                log.writelog('WARNING: Are you using real JWST data? If so, you should edit the flag_bg() function in nirspec.py and look at Issue #193 on Github!')
            elif meta.inst == 'niriss':
                raise ValueError('NIRISS observations are currently unsupported!')
            elif meta.inst == 'wfc3':
                from . import wfc3 as inst
                meta, log = inst.preparation_step(meta, log)
            else:
                raise ValueError('Unknown instrument {}'.format(meta.inst))

            stdspec = np.array([])
            # Loop over each segment
            # Only reduce the last segment/file if testing_S3 is set to True in ecf
            if meta.testing_S3:
                istart = meta.num_data_files - 1
            else:
                istart = 0
            for m in range(istart, meta.num_data_files):
                # Keep track if this is the first file - otherwise MIRI will keep swapping x and y windows
                if m==istart and meta.spec_hw==meta.spec_hw_range[0] and meta.bg_hw==meta.bg_hw_range[0]:
                    meta.firstFile = True
                else:
                    meta.firstFile = False
                # Report progress
                if meta.verbose:
                    log.writelog(f'Reading file {m + 1} of {meta.num_data_files}')
                else:
                    log.writelog(f'Reading file {m + 1} of {meta.num_data_files}', end='\r')

                # Read in data frame and header
                data, meta = inst.read(meta.segment_list[m], data, meta)

                # Get number of integrations and frame dimensions
                meta.n_int, meta.ny, meta.nx = data.data.shape
                if meta.testing_S3:
                    # Only process the last 5 integrations when testing
                    meta.int_start = np.max((0,meta.n_int-5))
                else:
                    meta.int_start = 0

                # Trim data to subarray region of interest
                data, meta = util.trim(data, meta)

                # Locate source postion
                meta.src_ypos = source_pos.source_pos(data, meta, m, header=('SRCYPOS' in data.shdr))
                log.writelog(f'  Source position on detector is row {meta.src_ypos}.', mute=(not meta.verbose))

                # Convert flux units to electrons (eg. MJy/sr -> DN -> Electrons)
                data, meta = b2f.convert_to_e(data, meta, log)

                # Create bad pixel mask (1 = good, 0 = bad)
                # FINDME: Will want to use DQ array in the future to flag certain pixels
                data.submask = np.ones(data.subdata.shape)

                # Check if arrays have NaNs
                data.submask = util.check_nans(data.subdata, data.submask, log, name='SUBDATA')
                data.submask = util.check_nans(data.suberr, data.submask, log, name='SUBERR')
                data.submask = util.check_nans(data.subv0, data.submask, log, name='SUBV0')

                # Manually mask regions [colstart, colend, rowstart, rowend]
                if hasattr(meta, 'manmask'):
                    log.writelog("  Masking manually identified bad pixels", mute=(not meta.verbose))
                    for i in range(len(meta.manmask)):
                        ind, colstart, colend, rowstart, rowend = meta.manmask[i]
                        data.submask[rowstart:rowend, colstart:colend] = 0

                # Perform outlier rejection of sky background along time axis
                log.writelog('  Performing background outlier rejection', mute=(not meta.verbose))
                meta.bg_y2 = int(meta.src_ypos + bg_hw_val)
                meta.bg_y1 = int(meta.src_ypos - bg_hw_val)
                data = inst.flag_bg(data, meta)

                data = bg.BGsubtraction(data, meta, log, meta.isplots_S3)

                if meta.isplots_S3 >= 3:
                    log.writelog('  Creating figures for background subtraction', mute=(not meta.verbose))
                    iterfn = range(meta.int_start,meta.n_int)
                    if meta.verbose:
                        iterfn = tqdm(iterfn)
                    for n in iterfn:
                        # make image+background plots
                        plots_s3.image_and_background(data, meta, n, m)

                # Calulate and correct for 2D drift
                if hasattr(inst, 'correct_drift2D'):
                    log.writelog('  Correcting for 2D drift', mute=(not meta.verbose))
                    inst.correct_drift2D(data, meta, m)

                # Select only aperture region
                ap_y1 = int(meta.src_ypos - spec_hw_val)
                ap_y2 = int(meta.src_ypos + spec_hw_val)
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
                log.writelog("  Performing optimal spectral extraction", mute=(not meta.verbose))
                data.optspec = np.zeros(data.stdspec.shape)
                data.opterr  = np.zeros(data.stdspec.shape)

                gain = 1  # Already converted DN to electrons, so gain = 1 for optspex
                iterfn = range(meta.int_start,meta.n_int)
                if meta.verbose:
                    iterfn = tqdm(iterfn)
                for n in iterfn:
                    data.optspec[n], data.opterr[n], mask = optspex.optimize(meta, data.apdata[n], data.apmask[n], data.apbg[n],
                                                                             data.stdspec[n], gain, data.apv0[n],
                                                                             p5thresh=meta.p5thresh, p7thresh=meta.p7thresh,
                                                                             fittype=meta.fittype, window_len=meta.window_len,
                                                                             deg=meta.prof_deg, n=data.intstart + n,
                                                                             isplots=meta.isplots_S3, meddata=data.medapdata)
                #Mask out NaNs
                data.optspec = np.ma.masked_invalid(data.optspec)
                data.opterr = np.ma.masked_invalid(data.opterr)
                mask = np.logical_or(np.ma.getmaskarray(data.optspec), np.ma.getmaskarray(data.opterr))
                data.optspec = np.ma.masked_where(mask, data.optspec)
                data.opterr = np.ma.masked_where(mask, data.opterr)
                # Plot results
                if meta.isplots_S3 >= 3:
                    log.writelog('  Creating figures for optimal spectral extraction', mute=(not meta.verbose))
                    iterfn = range(meta.int_start,meta.n_int)
                    if meta.verbose:
                        iterfn = tqdm(iterfn)
                    for n in iterfn:
                        # make optimal spectrum plot
                        plots_s3.optimal_spectrum(data, meta, n, m)

                # Append results
                if len(stdspec) == 0:
                    wave_2d = data.subwave
                    wave_1d = data.subwave[meta.src_ypos]
                    stdspec = data.stdspec
                    stdvar  = data.stdvar
                    optspec = data.optspec
                    opterr  = data.opterr
                    time    = data.time
                else:
                    stdspec = np.append(stdspec, data.stdspec, axis=0)
                    stdvar  = np.append(stdvar, data.stdvar, axis=0)
                    optspec = np.append(optspec, data.optspec, axis=0)
                    opterr  = np.append(opterr, data.opterr, axis=0)
                    time    = np.append(time, data.time, axis=0)

            if meta.inst == 'wfc3':
                # WFC3 needs a conclusion step to convert lists into arrays before saving
                meta, log = inst.conclusion_step(meta, log)

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            if meta.save_output == True:
                log.writelog('Saving results as astropy table')
                meta.tab_filename = meta.outputdir + 'S3_' + event_ap_bg + "_Table_Save.txt"
                astropytable.savetable_S3(meta.tab_filename, time, wave_1d, stdspec, stdvar, optspec, opterr)

            # Compute MAD alue
            meta.mad_s3 = util.get_mad(meta, wave_1d, optspec)
            log.writelog("Stage 3 MAD = " + str(np.round(meta.mad_s3, 2).astype(int)) + " ppm")

            if meta.isplots_S3 >= 1:
                log.writelog('Generating figure')
                # 2D light curve without drift correction
                plots_s3.lc_nodriftcorr(meta, wave_1d, optspec)

            # Save results
            if meta.save_output == True:
                log.writelog('Saving Metadata')
                me.saveevent(meta, meta.outputdir + 'S3_' + event_ap_bg + "_Meta_Save", save=[])

            log.closelog()

    return meta

def read_s2_meta(meta):
    '''Loads in an S2 meta file.

    Parameters
    ----------
    meta:    MetaClass
        The new meta object for the current S3 processing.

    Returns
    -------
    s2_meta:   MetaClass
        The S2 metadata object.

    Notes
    -------
    History:

    - March 2022 Taylor Bell
        Initial version.
    '''
    # Search for the S2 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    fnames = glob.glob(rootdir+'S2_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(fnames)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        fnames = glob.glob(rootdir+'**/S2_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)
        fnames = sn.sort_nicely(fnames)

    if len(fnames)>=1:
        # get the folder with the latest modified time
        fname = max(fnames, key=os.path.getmtime)

    if len(fnames)==0:
        # There may be no metafiles in the inputdir - raise an error and give a helpful message
        print('WARNING: Unable to find an output metadata file from Eureka!\'s S2 step '
                +'in the inputdir: \n"{}"!\n'.format(meta.inputdir)
                +'Assuming this S2 data was produced by the JWST pipeline instead.')
        return None
    elif len(fnames)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(fname)
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = fname[:-4] # Strip off the .dat ending

    s2_meta = me.loadevent(fname)

    # Code to not break backwards compatibility with old MetaClass save files but also use the new MetaClass going forwards
    s2_meta = readECF.MetaClass(**s2_meta.__dict__)

    return s2_meta

def load_general_s2_meta_info(meta, ecf_path, s2_meta):
    '''Loads in the S2 meta save file and adds in attributes from the S3 ECF.

    Parameters
    ----------
    meta:    MetaClass
        The new meta object for the current S3 processing.
    ecf_path:
        The absolute path to where the S3 ECF is stored.

    Returns
    -------
    meta:   MetaClass
        The S2 metadata object with attributes added by S3.

    Notes
    -------
    History:

    - March 2022 Taylor Bell
        Initial version.
    '''
    # Need to remove the topdir from the outputdir
    s2_outputdir = s2_meta.outputdir[len(meta.topdir):]
    if s2_outputdir[0]=='/':
        s2_outputdir = s2_outputdir[1:]
    if s2_outputdir[-1]!='/':
        s2_outputdir += '/'
    s2_topdir = s2_meta.topdir
    
    # Load S3 Eureka! control file and store values in the S2 metadata object
    ecffile = 'S3_' + meta.eventlabel + '.ecf'
    meta = s2_meta
    meta.read(ecf_path, ecffile)

    # Overwrite the inputdir with the exact output directory from S2
    meta.inputdir = os.path.join(s2_topdir, s2_outputdir)
    meta.old_datetime = meta.datetime # Capture the date that the S2 data was made (to figure out it's foldername)
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = s2_outputdir
    meta.outputdir_raw = meta.outputdir

    return meta
