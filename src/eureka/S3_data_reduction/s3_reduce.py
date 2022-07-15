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

import os
import time as time_pkg
import numpy as np
import astraeus.xarrayIO as xrio
from tqdm import tqdm
import psutil
from . import optspex
from . import plots_s3, source_pos
from . import background as bg
from . import bright2flux as b2f
from ..lib import logedit
from ..lib import readECF
from ..lib import manageevent as me
from ..lib import util


def reduce(eventlabel, ecf_path=None, s2_meta=None):
    '''Reduces data images and calculates optimal spectra.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
        Defaults to None which resolves to './'.
    s2_meta : eureka.lib.readECF.MetaClass; optional
        The metadata object from Eureka!'s S2 step (if running S2 and S3
        sequentially). Defaults to None.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S3.

    Notes
    -----
    History:

    - May 2021 Kevin Stevenson
        Initial version
    - October 2021 Taylor Bell
        Updated to allow for inputs from S2
    - July 2022 Caroline Piaulet
        Now computing the y pos and width for each integration 
        + stored in Spec and add diagnostics plots
    '''

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel
    meta.datetime = time_pkg.strftime('%Y-%m-%d')

    if s2_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into
        # that old MetaClass
        s2_meta, meta.inputdir, meta.inputdir_raw = \
            me.findevent(meta, 'S2', allowFail=True)
    else:
        # Running these stages sequentially, so can safely assume
        # the path hasn't changed
        meta.inputdir = s2_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]

    if s2_meta is None:
        # Attempt to find subdirectory containing S2 FITS files
        meta = util.find_fits(meta)
    else:
        meta = me.mergeevents(meta, s2_meta)

    # check for range of spectral apertures
    if isinstance(meta.spec_hw, list):
        meta.spec_hw_range = range(meta.spec_hw[0],
                                   meta.spec_hw[1]+meta.spec_hw[2],
                                   meta.spec_hw[2])
    else:
        meta.spec_hw_range = [meta.spec_hw]

    # check for range of background apertures
    if isinstance(meta.bg_hw, list):
        meta.bg_hw_range = range(meta.bg_hw[0],
                                 meta.bg_hw[1]+meta.bg_hw[2],
                                 meta.bg_hw[2])
    else:
        meta.bg_hw_range = [meta.bg_hw]

    # create directories to store data
    # run_s3 used to make sure we're always looking at the right run for
    # each aperture/annulus pair
    meta.run_s3 = None
    for spec_hw_val in meta.spec_hw_range:

        for bg_hw_val in meta.bg_hw_range:

            meta.eventlabel = eventlabel

            meta.run_s3 = util.makedirectory(meta, 'S3', meta.run_s3,
                                             ap=spec_hw_val, bg=bg_hw_val)

    # begin process
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            meta.outputdir = util.pathdirectory(meta, 'S3', meta.run_s3,
                                                ap=spec_hw_val, bg=bg_hw_val)

            event_ap_bg = (meta.eventlabel+"_ap"+str(spec_hw_val)+'_bg' +
                           str(bg_hw_val))

            # Open new log file
            meta.s3_logname = meta.outputdir + 'S3_' + event_ap_bg + ".log"
            if s2_meta is not None:
                log = logedit.Logedit(meta.s3_logname, read=s2_meta.s2_logname)
            else:
                log = logedit.Logedit(meta.s3_logname)
            log.writelog("\nStarting Stage 3 Reduction\n")
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")
            log.writelog(f"Using ap={spec_hw_val}, bg={bg_hw_val}")

            # Copy ecf
            log.writelog('Copying S3 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            # Create list of file segments
            meta = util.readfiles(meta, log)

            # Load instrument module
            if meta.inst == 'miri':
                from . import miri as inst
            elif meta.inst == 'nircam':
                from . import nircam as inst
            elif meta.inst == 'nirspec':
                from . import nirspec as inst
                log.writelog('WARNING: Are you using real JWST data? If so, '
                             'you should edit the flag_bg() function in '
                             'nirspec.py and look at Issue #193 on Github!')
            elif meta.inst == 'niriss':
                raise ValueError('NIRISS observations are currently '
                                 'unsupported!')
            elif meta.inst == 'wfc3':
                from . import wfc3 as inst
                meta, log = inst.preparation_step(meta, log)
            else:
                raise ValueError('Unknown instrument {}'.format(meta.inst))

            # Loop over each segment
            # Only reduce the last segment/file if testing_S3 is set to
            # True in ecf
            if meta.testing_S3:
                istart = meta.num_data_files - 1
            else:
                istart = 0

            # Group files into batches
            if not hasattr(meta, 'max_memory'):
                meta.max_memory = 0.5
            if not hasattr(meta, 'nfiles'):
                meta.nfiles = 1
            system_RAM = psutil.virtual_memory().total
            filesize = os.path.getsize(meta.segment_list[istart])
            maxfiles = max([1, int(system_RAM*meta.max_memory/filesize)])
            meta.files_per_batch = min([maxfiles, meta.nfiles])
            meta.nbatch = int(np.ceil((meta.num_data_files-istart) /
                                      meta.files_per_batch))

            datasets = []
            for m in range(meta.nbatch):
                first_file = m*meta.files_per_batch
                last_file = min([meta.num_data_files,
                                 (m+1)*meta.files_per_batch])
                nfiles = last_file-first_file
                               
                # Report progress
                if meta.files_per_batch > 1:
                    message = (f'Starting batch {m + 1} of {meta.nbatch} '
                               f'with {nfiles} files')
                else:
                    message = f'Starting file {m + 1} of {meta.num_data_files}'
                if meta.verbose:
                    log.writelog(message)
                else:
                    log.writelog(message, end='\r')

                # Read in data frame and header
                batch = []
                for i in range(first_file, last_file):
                    # Keep track if this is the first file - otherwise
                    # MIRI will keep swapping x and y windows
                    meta.firstFile = m == 0 and i == 0
                    meta.firstInBatch = i == 0
                    # Initialize a new data object
                    data = xrio.makeDataset()
                    data, meta, log = inst.read(meta.segment_list[i], data, 
                                                meta, log)
                    batch.append(data)

                # Combine individual datasets
                if meta.files_per_batch > 1:
                    log.writelog('  Concatenating files...',
                                 mute=(not meta.verbose))
                data = xrio.concat(batch)
                data.attrs['intstart'] = batch[0].attrs['intstart']
                data.attrs['intend'] = batch[-1].attrs['intend']

                # Create dataset to store y position and width
                if meta.record_ypos:
                    src_ypos_exact = np.zeros_like(data["time"])
                    src_ypos_width = np.zeros_like(data["time"])

                # Get number of integrations and frame dimensions
                meta.n_int, meta.ny, meta.nx = data.flux.shape
                if meta.testing_S3:
                    # Only process the last 5 integrations when testing
                    meta.int_start = np.max((0, meta.n_int-5))
                else:
                    meta.int_start = 0

                # Trim data to subarray region of interest
                # Dataset object no longer contains untrimmed data
                data, meta = util.trim(data, meta)

                # Create bad pixel mask (1 = good, 0 = bad)
                # FINDME: Will want to use DQ array in the future
                # to flag certain pixels
                data['mask'] = (['time', 'y', 'x'],
                                np.ones(data.flux.shape, dtype=bool))

                # Check if arrays have NaNs
                log.writelog('  Masking NaNs in data arrays...',
                             mute=(not meta.verbose))
                data.mask.values = util.check_nans(data.flux.values,
                                                   data.mask.values,
                                                   log, name='FLUX')
                data.mask.values = util.check_nans(data.err.values,
                                                   data.mask.values,
                                                   log, name='ERR')
                data.mask.values = util.check_nans(data.v0.values,
                                                   data.mask.values,
                                                   log, name='V0')

                # Manually mask regions [colstart, colend, rowstart, rowend]
                if hasattr(meta, 'manmask'):
                    data = util.manmask(data, meta, log)

                # Locate source postion
                log.writelog('  Locating source position...',
                             mute=(not meta.verbose))
                meta.src_ypos, _, _ = source_pos.source_pos(data, meta, m)
                log.writelog(f'    Source position on detector is row '
                             f'{meta.src_ypos}.', mute=(not meta.verbose))

                # Compute 1D wavelength solution
                if 'wave_2d' in data:
                    data['wave_1d'] = (['x'],
                                       data.wave_2d[meta.src_ypos].values)
                    data['wave_1d'].attrs['wave_units'] = \
                        data.wave_2d.attrs['wave_units']

                # Convert flux units to electrons
                # (eg. MJy/sr -> DN -> Electrons)
                data, meta = b2f.convert_to_e(data, meta, log)

                # Compute median frame
                data['medflux'] = (['y', 'x'], np.median(data.flux.values,
                                                         axis=0))
                data['medflux'].attrs['flux_units'] = \
                    data.flux.attrs['flux_units']

                # correct G395H curvature
                if meta.inst == 'nirspec' and data.mhdr['GRATING'] == 'G395H':
                    if meta.curvature == 'correct':
                        log.writelog('  Correcting for G395H curvature...',
                                     mute=(not meta.verbose))
                        data, meta = inst.straighten_trace(data, meta, log)

                # Perform outlier rejection of sky background along time axis
                data = inst.flag_bg(data, meta, log)

                # Do the background subtraction
                data = bg.BGsubtraction(data, meta, log, meta.isplots_S3)

                # Make image+background plots
                if meta.isplots_S3 >= 3:
                    plots_s3.image_and_background(data, meta, log, m)

                # Calulate and correct for 2D drift
                if hasattr(inst, 'correct_drift2D'):
                    data, meta, log = inst.correct_drift2D(data, meta, log, m)

                # Select only aperture region
                apdata, aperr, apmask, apbg, apv0 = inst.cut_aperture(data,
                                                                      meta,
                                                                      log)

                # Extract standard spectrum and its variance
                data = optspex.standard_spectrum(data, apdata, aperr)

                # Extract optimal spectrum with uncertainties
                log.writelog("  Performing optimal spectral extraction...",
                             mute=(not meta.verbose))
                data['optspec'] = (['time', 'x'], np.zeros(data.stdspec.shape))
                data['opterr'] = (['time', 'x'], np.zeros(data.stdspec.shape))
                data['optspec'].attrs['flux_units'] = \
                    data.flux.attrs['flux_units']
                data['optspec'].attrs['time_units'] = \
                    data.flux.attrs['time_units']
                data['opterr'].attrs['flux_units'] = \
                    data.flux.attrs['flux_units']
                data['opterr'].attrs['time_units'] = \
                    data.flux.attrs['time_units']

                # Compute median frame
                medapdata = np.median(apdata, axis=0)
                # Already converted DN to electrons, so gain = 1 for optspex
                gain = 1
                iterfn = range(meta.int_start, meta.n_int)
                if meta.verbose:
                    iterfn = tqdm(iterfn)
                for n in iterfn:
                    # when loop over ints, get exact y pos and width
                    if meta.record_ypos:
                        src_pos_results = source_pos.source_pos(data, meta,
                                                                m, n)
                        _, ypos_exact, ypos_width = src_pos_results
                        src_ypos_exact[n] = ypos_exact
                        src_ypos_width[n] = ypos_width

                    data['optspec'][n], data['opterr'][n], mask = \
                        optspex.optimize(meta, apdata[n], apmask[n], apbg[n],
                                         data.stdspec[n].values, gain, apv0[n],
                                         p5thresh=meta.p5thresh,
                                         p7thresh=meta.p7thresh,
                                         fittype=meta.fittype,
                                         window_len=meta.window_len,
                                         deg=meta.prof_deg, n=n, m=m,
                                         meddata=medapdata)

                # Mask out NaNs and Infs
                optspec_ma = np.ma.masked_invalid(data.optspec.values)
                opterr_ma = np.ma.masked_invalid(data.opterr.values)
                optmask = np.logical_or(np.ma.getmaskarray(optspec_ma),
                                        np.ma.getmaskarray(opterr_ma))
                data['optmask'] = (['time', 'x'], optmask)

                if meta.record_ypos:
                    data['driftypos'] = (['time'], src_ypos_exact)
                    data['driftywidth'] = (['time'], src_ypos_width)

                # Plot results
                if meta.isplots_S3 >= 3:
                    log.writelog('  Creating figures for optimal spectral '
                                 'extraction', mute=(not meta.verbose))
                    iterfn = range(meta.int_start, meta.n_int)
                    if meta.verbose:
                        iterfn = tqdm(iterfn)
                    for n in iterfn:
                        # make optimal spectrum plot
                        plots_s3.optimal_spectrum(data, meta, n, m)
                    if meta.record_ypos:
                        # make y position and width plots
                        plots_s3.driftypos(data, meta)
                        plots_s3.driftywidth(data, meta)

                if meta.save_output:
                    # Save flux data from current segment
                    filename_xr = (meta.outputdir+'S3_'+event_ap_bg +
                                   "_FluxData_seg"+str(m).zfill(4)+".h5")
                    success = xrio.writeXR(filename_xr, data, verbose=False,
                                           append=False)
                    if success == 0:
                        del(data.attrs['filename'])
                        del(data.attrs['mhdr'])
                        del(data.attrs['shdr'])
                        success = xrio.writeXR(filename_xr, data,
                                               verbose=meta.verbose,
                                               append=False)

                # Remove large 3D arrays from Dataset
                del(data['flux'], data['err'], data['dq'], data['v0'],
                    data['bg'], data['mask'], data['wave_2d'],
                    data.attrs['intstart'], data.attrs['intend'])
                if meta.inst == 'wfc3':
                    del(data['flatmask'], data['variance'])

                # Append results for future concatenation
                datasets.append(data)

            # Concatenate results along time axis (default)
            spec = xrio.concat(datasets)

            # Plot fitted 2D drift
            # Note: This needs to happen before calling conclusion_step()
            if meta.isplots_S3 >= 1 and meta.inst == 'wfc3':
                plots_s3.drift_2D(spec, meta)

            if meta.inst == 'wfc3':
                # WFC3 needs a conclusion step to convert lists into
                # arrays before saving
                spec, meta, log = inst.conclusion_step(spec, meta, log)

            # Save Dataset object containing time-series of 1D spectra
            meta.filename_S3_SpecData = (meta.outputdir+'S3_'+event_ap_bg +
                                         "_SpecData.h5")
            success = xrio.writeXR(meta.filename_S3_SpecData, spec,
                                   verbose=True)

            # Compute MAD value
            meta.mad_s3 = util.get_mad(meta, log, spec.wave_1d, spec.optspec,
                                       optmask=spec.optmask)
            log.writelog(f"Stage 3 MAD = {int(np.round(meta.mad_s3))} ppm")

            if meta.isplots_S3 >= 1:
                log.writelog('Generating figure')
                # 2D light curve without drift correction
                plots_s3.lc_nodriftcorr(meta, spec.wave_1d, spec.optspec,
                                        optmask=spec.optmask)

            # Save results
            if meta.save_output:
                log.writelog('Saving Metadata')
                fname = meta.outputdir + 'S3_' + event_ap_bg + "_Meta_Save"
                me.saveevent(meta, fname, save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return spec, meta
