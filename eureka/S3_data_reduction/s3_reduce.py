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


import os, time, glob
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from . import optspex
from . import plots_s3, source_pos
from . import background as bg
from . import bright2flux as b2f
from ..lib import sort_nicely as sn
from ..lib import logedit
from ..lib import readECF as rd
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


def reduceJWST(eventlabel, s2_meta=None):
    '''Reduces data images and calculates optimal spectra.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
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

    # Initialize a new metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)
    meta.eventlabel=eventlabel

    # S3 is not being called right after S2 - try to load a metadata in case S2 was previously run
    if s2_meta == None:
        # Search for the S2 output metadata in the inputdir provided in
        rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
        if rootdir[-1]!='/':
            rootdir += '/'
        fnames = glob.glob(rootdir+'**/S2_'+meta.eventlabel+'_Meta_Save.dat', recursive=True)
        fnames = sn.sort_nicely(fnames)

        if len(fnames)==0:
            # There may be no metafiles in the inputdir - raise an error and give a helpful message
            print('WARNING: Unable to find an output metadata file from Eureka!\'s S2 step '
                 +'in the inputdir: \n"{}"!\n'.format(meta.inputdir)
                 +'Assuming this S2 data was produced by the JWST pipeline instead.')
        else:
            if len(fnames)>1:
                # There may be multiple runs - use the most recent but warn the user
                print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(meta.inputdir)
                     +'Using the metadata file: \n"{}"'.format(fnames[-1]))

            fname = fnames[-1] # Pick the last file name
            fname = fname[:-4] # Strip off the .dat ending

            s2_meta = me.loadevent(fname)

    # Locate the exact output folder from the previous S2 run (since there is a procedurally generated subdirectory for each run)
    if s2_meta != None:
        # Need to remove the topdir from the outputdir
        if os.path.isdir(s2_meta.outputdir):
            s2_outputdir = s2_meta.outputdir[len(s2_meta.topdir):]
            if s2_outputdir[0]=='/':
                s2_outputdir = s2_outputdir[1:]

            meta = s2_meta

            # Load Eureka! control file and store values in the S2 metadata object
            ecffile = 'S3_' + eventlabel + '.ecf'
            ecf = rd.read_ecf(ecffile)
            rd.store_ecf(meta, ecf)

            # Overwrite the inputdir with the exact output directory from S2
            meta.inputdir = s2_outputdir
        else:
            raise AssertionError("Unable to find output data files from Eureka!'s S2 step! "
                                 + "Looked in the folder: \n{}".format(s2_meta.outputdir))

    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir

    # check for range of spectral apertures
    if isinstance(meta.spec_hw, list):
        meta.spec_hw_range = range(meta.spec_hw[0], meta.spec_hw[1]+meta.spec_hw[2], meta.spec_hw[2])
    else:
        meta.spec_hw_range = [meta.spec_hw]

    #check for range of background apertures
    if isinstance(meta.bg_hw, list):
        meta.bg_hw_range = range(meta.bg_hw[0], meta.bg_hw[1]+meta.spec_hw[2], meta.bg_hw[2])
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

            t0 = time.time()

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

            # Copy ecf (and update inputdir in case S3 is being called sequentially with S2)
            log.writelog('Copying S3 control file')
            new_ecfname = meta.outputdir + ecffile.split('/')[-1]
            with open(new_ecfname, 'w') as new_file:
                with open(ecffile, 'r') as file:
                    for line in file.readlines():
                        if len(line.strip())==0 or line.strip()[0]=='#':
                            new_file.write(line)
                        else:
                            line_segs = line.strip().split()
                            if line_segs[0]=='inputdir':
                                new_file.write(line_segs[0]+'\t\t/'+meta.inputdir+'\t'+' '.join(line_segs[2:])+'\n')
                            else:
                                new_file.write(line)

            # Create list of file segments
            meta = util.readfiles(meta)
            num_data_files = len(meta.segment_list)
            if num_data_files==0:
                rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
                if rootdir[-1]!='/':
                    rootdir += '/'
                raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{rootdir}"!')
            else:
                log.writelog(f'\nFound {num_data_files} data file(s) ending in {meta.suffix}.fits')

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
            elif meta.inst == 'niriss':
                raise ValueError('NIRISS observations are currently unsupported!')
            else:
                raise ValueError('Unknown instrument {}'.format(meta.inst))

            stdspec = np.array([])
            # Loop over each segment
            # Only reduce the last segment/file if testing_S3 is set to True in ecf
            if meta.testing_S3:
                istart = num_data_files - 1
            else:
                istart = 0
            for m in range(istart, num_data_files):
                # Keep track if this is the first file - otherwise MIRI will keep swapping x and y windows
                if m==istart:
                    meta.firstFile = True
                else:
                    meta.firstFile = False
                # Report progress
                log.writelog(f'Reading file {m + 1} of {num_data_files}')
                # Read in data frame and header
                data, meta = inst.read(meta.segment_list[m], data, meta)
                # Get number of integrations and frame dimensions
                meta.n_int, meta.ny, meta.nx = data.data.shape
                if meta.testing_S3:
                    # Only process the last 5 integrations when testing
                    meta.int_start = np.max((0,meta.n_int-5))
                else:
                    meta.int_start = 0
                # Locate source postion
                meta.src_ypos = source_pos.source_pos(data, meta, m, header=('SRCYPOS' in data.shdr))
                log.writelog(f'  Source position on detector is row {meta.src_ypos}.')
                # Trim data to subarray region of interest
                data, meta = util.trim(data, meta)
                # Create bad pixel mask (1 = good, 0 = bad)
                # FINDME: Will want to use DQ array in the future to flag certain pixels
                data.submask = np.ones(data.subdata.shape)

                # Convert flux units to electrons (eg. MJy/sr -> DN -> Electrons)
                data, meta = b2f.convert_to_e(data, meta, log)

                # Check if arrays have NaNs
                data.submask = util.check_nans(data.subdata, data.submask, log, name='SUBDATA')
                data.submask = util.check_nans(data.suberr, data.submask, log, name='SUBERR')
                data.submask = util.check_nans(data.subv0, data.submask, log, name='SUBV0')

                # Manually mask regions [colstart, colend, rowstart, rowend]
                if hasattr(meta, 'manmask'):
                    log.writelog("  Masking manually identified bad pixels")
                    for i in range(len(meta.manmask)):
                        ind, colstart, colend, rowstart, rowend = meta.manmask[i]
                        data.submask[rowstart:rowend, colstart:colend] = 0

                # Perform outlier rejection of sky background along time axis
                log.writelog('  Performing background outlier rejection')
                meta.bg_y2 = int(meta.src_ypos + bg_hw_val)
                meta.bg_y1 = int(meta.src_ypos - bg_hw_val)
                data = inst.flag_bg(data, meta)

                data = bg.BGsubtraction(data, meta, log, meta.isplots_S3)

                # Calulate drift2D
                # print("Calculating 2D drift...")

                # print("Performing rough, pixel-scale drift correction...")

                # Outlier rejection of full frame along time axis
                # print("Performing full-frame outlier rejection...")

                if meta.isplots_S3 >= 3:
                    log.writelog('  Creating figures for background subtraction')
                    for n in tqdm(range(meta.int_start,meta.n_int)):
                        # make image+background plots
                        plots_s3.image_and_background(data, meta, n)

                # print("Performing sub-pixel drift correction...")

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
                log.writelog("  Performing optimal spectral extraction")
                data.optspec = np.zeros(data.stdspec.shape)
                data.opterr  = np.zeros(data.stdspec.shape)
                gain = 1  # Already converted DN to electrons, so gain = 1 for optspex
                for n in tqdm(range(meta.int_start,meta.n_int)):
                    data.optspec[n], data.opterr[n], mask = optspex.optimize(data.apdata[n], data.apmask[n], data.apbg[n],
                                                                             data.stdspec[n], gain, data.apv0[n],
                                                                             p5thresh=meta.p5thresh, p7thresh=meta.p7thresh,
                                                                             fittype=meta.fittype, window_len=meta.window_len,
                                                                             deg=meta.prof_deg, n=data.intstart + n,
                                                                             isplots=meta.isplots_S3, eventdir=meta.outputdir,
                                                                             meddata=data.medapdata, hide_plots=meta.hide_plots)

                # Plot results
                if meta.isplots_S3 >= 3:
                    log.writelog('  Creating figures for optimal spectral extraction')
                    for n in tqdm(range(meta.int_start,meta.n_int)):
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

            if meta.save_output == True:
                log.writelog('Saving results as astropy table')
                meta.tab_filename = meta.outputdir + 'S3_' + event_ap_bg + "_Table_Save.txt"
                astropytable.savetable_S3(meta.tab_filename, bjdtdb, wave_1d, stdspec, stdvar, optspec, opterr)

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
