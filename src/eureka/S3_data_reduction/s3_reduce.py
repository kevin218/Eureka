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
from copy import deepcopy
import astraeus.xarrayIO as xrio
from tqdm import tqdm
import psutil

from . import optspex
from . import plots_s3, source_pos, straighten
from . import background as bg
from . import bright2flux as b2f

from .s3_meta import S3MetaClass
from ..lib import logedit
from ..lib import manageevent as me
from ..lib import util
from ..lib import centerdriver, apphot


def reduce(eventlabel, ecf_path=None, s2_meta=None, input_meta=None):
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
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

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
    - July 2022 Sebastian Zieba
        Added photometry S3
    - Feb 2023 Isaac Edelman
        Added new centroiding method (mgmc_pri, mgmc_sec) to
        correct for shortwave photometry data processing issues
    '''
    s2_meta = deepcopy(s2_meta)
    input_meta = deepcopy(input_meta)

    if input_meta is None:
        meta = S3MetaClass(folder=ecf_path, eventlabel=eventlabel)
    else:
        meta = S3MetaClass(**input_meta.__dict__)

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
        meta = S3MetaClass(**me.mergeevents(meta, s2_meta).__dict__)

    # Create list of file segments
    meta = util.readfiles(meta)

    # First apply any instrument-specific defaults
    if meta.photometry:
        if meta.inst == 'miri':
            meta.set_MIRI_Photometry_defaults()
        elif meta.inst == 'nircam':
            meta.set_NIRCam_Photometry_defaults()
    else:
        if meta.inst == 'miri':
            meta.set_MIRI_defaults()
        elif meta.inst == 'nircam':
            meta.set_NIRCam_defaults()
        elif meta.inst == 'nirspec':
            meta.set_NIRSpec_defaults()
        elif meta.inst == 'niriss':
            meta.set_NIRISS_defaults()
        elif meta.inst == 'wfc3':
            meta.set_WFC3_defaults()
    # Then apply instrument-agnostic defaults
    meta.set_defaults()

    # Setup range of spectral apertures
    meta.setup_aperture_radii()

    # create directories to store data
    # run_s3 used to make sure we're always looking at the right run for
    # each aperture/annulus pair
    meta.run_s3 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            meta.eventlabel = eventlabel
            if not isinstance(bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                bg_hw_val //= meta.expand
            meta.run_s3 = util.makedirectory(meta, 'S3', meta.run_s3,
                                             ap=spec_hw_val//meta.expand,
                                             bg=bg_hw_val)

    # begin process
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val
            # Directory structure should not use expanded HW values
            spec_hw_val //= meta.expand
            if not isinstance(bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                bg_hw_val //= meta.expand
            meta.outputdir = util.pathdirectory(meta, 'S3', meta.run_s3,
                                                ap=spec_hw_val,
                                                bg=bg_hw_val)

            event_ap_bg = (meta.eventlabel+"_ap"+str(spec_hw_val) +
                           '_bg' + str(bg_hw_val))

            # Open new log file
            meta.s3_logname = meta.outputdir + 'S3_' + event_ap_bg + ".log"
            if s2_meta is not None:
                log = logedit.Logedit(meta.s3_logname, read=s2_meta.s2_logname)
            else:
                log = logedit.Logedit(meta.s3_logname)
            log.writelog("\nStarting Stage 3 Reduction\n")
            log.writelog(f"Eureka! Version: {meta.version}", mute=True)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f'  Found {meta.num_data_files} data file(s) ending '
                         f'in {meta.suffix}.fits', mute=(not meta.verbose))
            log.writelog(f"Output directory: {meta.outputdir}")
            log.writelog(f"Using ap={spec_hw_val}, " +
                         f"bg={bg_hw_val}, " +
                         f"expand={meta.expand}")

            log.writelog(f"CRDS Context pmap: {meta.pmap}", mute=True)

            # Load instrument module
            if meta.inst == 'miri':
                from . import miri as inst
            elif meta.inst == 'nircam':
                from . import nircam as inst
            elif meta.inst == 'nirspec':
                from . import nirspec as inst
            elif meta.inst == 'niriss':
                raise ValueError('NIRISS observations are currently '
                                 'unsupported!')
            elif meta.inst == 'wfc3':
                from . import wfc3 as inst
                meta.bg_dir = 'CxC'
                meta, log = inst.preparation_step(meta, log)
            else:
                raise ValueError('Unknown instrument {}'.format(meta.inst))

            # Copy ecf
            log.writelog('Copying S3 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            # Loop over each segment
            # Only reduce the last segment/file if testing_S3 is set to
            # True in ecf
            if meta.testing_S3:
                istart = meta.num_data_files - 1
            else:
                istart = 0

            # Group files into batches
            if meta.nfiles == 1 and meta.nfiles > 1 and meta.indep_batches:
                log.writelog('WARNING: You have selected non-ideal settings '
                             'with indep_batches = True and nfiles = 1.'
                             'If your computer has enough RAM to '
                             'load many/all of your Stage 2 files, it is '
                             'strongly recommended to increase nfiles.')
            system_RAM = psutil.virtual_memory().total
            filesize = os.path.getsize(meta.segment_list[istart])*meta.expand
            maxfiles = max([1, int(system_RAM*meta.max_memory/filesize)])
            meta.files_per_batch = min([maxfiles, meta.nfiles])
            meta.nbatch = int(np.ceil((meta.num_data_files-istart) /
                                      meta.files_per_batch))

            datasets = []

            saved_refrence_tilt_frame = None
            saved_ref_median_frame = None

            for m in range(meta.nbatch):
                # Reset saved median frame if meta.indep_batches
                if meta.indep_batches:
                    saved_ref_median_frame = None
                    saved_refrence_tilt_frame = None

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
                    if i == last_file-1:
                        end = '\n'
                    else:
                        end = '\r'
                    # Initialize a new data object
                    data = xrio.makeDataset()
                    log.writelog(f'  Reading file {i+1}...',
                                 mute=(not meta.verbose), end=end)
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

                # Get number of integrations and frame dimensions
                meta.n_int, meta.ny, meta.nx = data.flux.shape
                if meta.testing_S3:
                    # Only process the last 5 integrations when testing
                    meta.int_start = np.max((0, meta.n_int-5))
                else:
                    meta.int_start = 0
                if meta.nplots is None:
                    meta.nplots = meta.n_int
                    meta.int_end = meta.n_int
                elif meta.int_start+meta.nplots > meta.n_int:
                    # Too many figures requested, so reduce it
                    meta.int_end = meta.n_int
                else:
                    meta.int_end = meta.int_start+meta.nplots

                # Initialize bad pixel mask (False = good, True = bad)
                data['mask'] = (['time', 'y', 'x'],
                                np.zeros(data.flux.shape, dtype=bool))

                # Perform BG subtraction along dispersion direction
                # for untrimmed NIRCam spectroscopic data
                if meta.bg_row_by_row:
                    meta.bg_dir = 'RxR'
                    data = bg.BGsubtraction(data, meta, log,
                                            m, meta.isplots_S3)
                    meta.bg_row_by_row = False
                    meta.bg_deg = None
                # Specify direction = CxC to perform standard BG subtraction
                # later on in Stage 3. This needs to be set independent of
                # having performed RxR BG subtraction.
                meta.bg_dir = 'CxC'

                # Trim data to subarray region of interest
                # Dataset object no longer contains untrimmed data
                data, meta = util.trim(data, meta)

                # Check if arrays have NaNs/infs
                log.writelog('  Masking NaNs/infs in data arrays...',
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

                # Start masking pixels based on DQ flags
                # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags
                # Odd numbers in DQ array are bad pixels. Do not use.
                if meta.dqmask:
                    dqmask = np.where(data.dq.values % 2 == 1)
                    data.mask.values[dqmask] = True

                # Manually mask regions [colstart, colend, rowstart, rowend]
                if meta.manmask is not None:
                    data = util.manmask(data, meta, log)

                if not meta.photometry:
                    # Locate source postion for the first integration of
                    # the first batch
                    if meta.indep_batches or meta.src_ypos is None:
                        data, meta, log = \
                            source_pos.source_pos_wrapper(data, meta, log, m)

                    # Compute 1D wavelength solution
                    if 'wave_2d' in data:
                        data['wave_1d'] = (['x'],
                                           data.wave_2d[meta.src_ypos].values)
                        data['wave_1d'].attrs['wave_units'] = \
                            data.wave_2d.attrs['wave_units']

                    # Check for bad wavelengths (beyond wavelength solution)
                    util.check_nans(data.wave_1d.values, np.zeros(meta.subnx),
                                    log, name='wavelength')

                    if meta.calibrated_spectra:
                        # Instrument-specific steps for generating
                        # calibrated stellar spectra
                        data = inst.calibrated_spectra(data, meta, log)
                    else:
                        # Convert flux units to electrons
                        # (eg. MJy/sr -> DN -> Electrons)
                        data, meta = b2f.convert_to_e(data, meta, log)

                    # Perform outlier rejection of
                    # full frame along time axis
                    if meta.ff_outlier:
                        data = inst.flag_ff(data, meta, log)

                    if saved_ref_median_frame is None:
                        # Compute clean median frame
                        data = optspex.clean_median_flux(data, meta, log, m)
                        # Save the original median frame
                        saved_ref_median_frame = deepcopy(data.medflux)
                    else:
                        # Load the original median frame
                        data['medflux'] = saved_ref_median_frame

                    # correct spectral curvature
                    if meta.curvature == 'correct':
                        data, meta = straighten.straighten_trace(data, meta,
                                                                 log, m)
                    elif meta.inst == 'nirspec' and meta.filter != 'PRISM':
                        log.writelog('WARNING: NIRSpec GRISM spectra is '
                                     'significantly curved and will very '
                                     'likely benefit from setting '
                                     'meta.curvature to "correct".')
                    elif meta.inst == 'nircam':
                        log.writelog('WARNING: NIRCam spectra is slightly '
                                     'curved and may benefit from setting '
                                     'meta.curvature to "correct".')

                    # Perform outlier rejection of
                    # sky background along time axis
                    meta.bg_y2 = meta.src_ypos + meta.bg_hw + 1
                    meta.bg_y1 = meta.src_ypos - meta.bg_hw
                    if not meta.ff_outlier:
                        data = inst.flag_bg(data, meta, log)

                    # Do the background subtraction
                    data = bg.BGsubtraction(data, meta, log,
                                            m, meta.isplots_S3)

                    # Calulate and correct for 2D drift
                    if hasattr(inst, 'correct_drift2D'):
                        data, meta, log = inst.correct_drift2D(data, meta, log,
                                                               m)
                    elif meta.record_ypos:
                        # Record y position and width for all integrations
                        data, meta, log = \
                            source_pos.source_pos_wrapper(data, meta, log,
                                                          m, integ=None)
                        if meta.isplots_S3 >= 1:
                            # make y position and width plots
                            plots_s3.driftypos(data, meta, m)
                            plots_s3.driftywidth(data, meta, m)

                    # Select only aperture region
                    apdata, aperr, apmask, apbg, apv0 = \
                        inst.cut_aperture(data, meta, log)

                    # Extract standard spectrum and its variance
                    data = optspex.standard_spectrum(data, apdata, apmask,
                                                     aperr)

                    # Perform optimal extraction
                    data, meta, log = optspex.optimize_wrapper(data, meta, log,
                                                               apdata, apmask,
                                                               apbg, apv0, m=m)

                    # Plot results
                    if meta.isplots_S3 >= 3:
                        log.writelog('  Creating figures for optimal spectral '
                                     'extraction', mute=(not meta.verbose))
                        iterfn = range(meta.int_start, meta.int_end)
                        if meta.verbose:
                            iterfn = tqdm(iterfn)
                        for n in iterfn:
                            # make optimal spectrum plot
                            plots_s3.optimal_spectrum(data, meta, n, m)
                        if meta.inst != 'wfc3':
                            plots_s3.residualBackground(data, meta, m)

                else:  # Do Photometry reduction
                    meta.photap = meta.spec_hw
                    meta.skyin, meta.skyout = np.array(meta.bg_hw.split('_')
                                                       ).astype(int)

                    if meta.calibrated_spectra:
                        # Instrument-specific steps for generating
                        # calibrated stellar spectra
                        data = inst.calibrated_spectra(data, meta, log)
                    else:
                        # Convert flux units to electrons
                        # (eg. MJy/sr -> DN -> Electrons)
                        data, meta = b2f.convert_to_e(data, meta, log)

                    # Do outlier reduction along time axis for
                    # each individual pixel
                    if meta.flag_bg:
                        data = inst.flag_bg_phot(data, meta, log)

                    # Setting up arrays for photometry reduction
                    data = util.phot_arrays(data)

                    # Compute the median frame
                    # and position of first centroid guess
                    # for mgmc method
                    if meta.ctr_guess is not None:
                        guess = np.array(meta.ctr_guess)[::-1]
                        trim = np.array([meta.ywindow[0], meta.xwindow[0]])
                        position_pri = guess - trim
                    elif meta.centroid_method == 'mgmc':
                        position_pri, extra, refrence_median_frame = \
                            centerdriver.centerdriver(
                                'mgmc_pri', data.flux.values, guess=1, trim=0,
                                radius=None, size=None, meta=meta, i=None,
                                m=None, mask=data.mask.values,
                                saved_ref_median_frame=saved_ref_median_frame)
                        if saved_ref_median_frame is None:
                            saved_ref_median_frame = refrence_median_frame

                    # for loop for integrations
                    for i in tqdm(range(len(data.time)),
                                  desc='  Looping over Integrations'):
                        if (meta.isplots_S3 >= 3
                                and meta.oneoverf_corr is not None):
                            # save current flux into an array for
                            # plotting 1/f correction comparison
                            flux_w_oneoverf = np.copy(data.flux.values[i])

                        # Determine centroid position
                        # We do this twice. First a coarse estimation,
                        # then a more precise one.
                        # Use the center of the frame as an initial guess
                        if (meta.centroid_method == 'fgc' and
                                meta.ctr_guess is None):
                            centroid_guess = [data.flux.shape[1]//2,
                                              data.flux.shape[2]//2]
                            # Do a 2D gaussian fit to the whole frame
                            position_pri, extra = \
                                centerdriver.centerdriver(
                                    'fgc', data.flux.values[i], centroid_guess,
                                    0, 0, 0, mask=data.mask.values[i],
                                    uncd=None, fitbg=1, maskstar=True,
                                    expand=1.0, psf=None, psfctr=None, i=i,
                                    m=m, meta=meta)

                        if meta.oneoverf_corr is not None:
                            # Correct for 1/f
                            data = \
                                inst.do_oneoverf_corr(data, meta, i,
                                                      position_pri[1], log)
                            if meta.isplots_S3 >= 3 and i < meta.nplots:
                                plots_s3.phot_2d_frame_oneoverf(
                                    data, meta, m, i, flux_w_oneoverf)

                        # Use the determined centroid and
                        # cut out ctr_cutout_size pixels around it
                        # Then perform another 2D gaussian fit
                        position, extra, refrence_median_frame = \
                            centerdriver.centerdriver(
                                meta.centroid_method+'_sec',
                                data.flux.values[i],
                                guess=position_pri,
                                trim=meta.ctr_cutout_size,
                                radius=0, size=0,
                                mask=data.mask.values[i],
                                uncd=None, fitbg=1,
                                maskstar=True, expand=1, psf=None,
                                psfctr=None, i=i, m=m, meta=meta,
                                saved_ref_median_frame=saved_ref_median_frame)

                        # Store centroid positions and
                        # the Gaussian 1-sigma half-widths
                        data['centroid_y'][i], data['centroid_x'][i] = position
                        data['centroid_sy'][i], data['centroid_sx'][i] = extra

                        # Plot 2D frame, the centroid and the centroid position
                        if meta.isplots_S3 >= 3 and i < meta.nplots:
                            plots_s3.phot_2d_frame(data, meta, m, i)

                        # Interpolate masked pixels before we perform
                        # aperture photometry
                        if meta.interp_method is not None:
                            util.interp_masked(data, meta, i, log)

                        # Calculate flux in aperture and subtract
                        # background flux
                        aphot = apphot.apphot(
                            meta, image=data.flux[i].values,
                            ctr=position, photap=meta.photap,
                            skyin=meta.skyin, skyout=meta.skyout,
                            betahw=1, targpos=position,
                            mask=data.mask[i].values,
                            imerr=data.err[i].values,
                            skyfrac=0.1, med=True, expand=1,
                            isbeta=False, nochecks=False,
                            aperr=True, nappix=True, skylev=True,
                            skyerr=True, nskypix=True,
                            nskyideal=True, status=True,
                            betaper=True, aperture_shape=meta.aperture_shape)
                        # Save results into arrays
                        (data['aplev'][i], data['aperr'][i],
                            data['nappix'][i], data['skylev'][i],
                            data['skyerr'][i], data['nskypix'][i],
                            data['nskyideal'][i], data['status'][i],
                            data['betaper'][i]) = aphot

                # plot tilt events
                if meta.isplots_S3 >= 5 and meta.inst == 'nircam' and \
                   meta.photometry:
                    refrence_tilt_frame = \
                        plots_s3.tilt_events(meta, data, log, m,
                                             position,
                                             saved_refrence_tilt_frame)

                    if saved_refrence_tilt_frame is not None:
                        saved_refrence_tilt_frame = refrence_tilt_frame

                if meta.save_fluxdata:
                    # Save Meta information to attributes of Xarray
                    util.add_meta_to_xarray(meta, data)

                    # Save flux data from current segment
                    filename_xr = (meta.outputdir+'S3_'+event_ap_bg +
                                   "_FluxData_seg"+str(m).zfill(4)+".h5")
                    success = xrio.writeXR(filename_xr, data, verbose=False,
                                           append=False)
                    if success == 0:
                        del (data.attrs['filename'])
                        del (data.attrs['mhdr'])
                        del (data.attrs['shdr'])
                        success = xrio.writeXR(filename_xr, data,
                                               verbose=meta.verbose,
                                               append=False)
                    else:
                        print(f"Finished writing to {filename_xr}")
                else:
                    del (data.attrs['filename'])
                    del (data.attrs['mhdr'])
                    del (data.attrs['shdr'])

                # Remove large 3D arrays from Dataset
                del (data['err'], data['dq'], data['v0'],
                     data['mask'], data.attrs['intstart'],
                     data.attrs['intend'])
                if not meta.photometry:
                    del (data['flux'], data['bg'], data['wave_2d'])
                elif meta.inst == 'wfc3':
                    del (data['flatmask'], data['variance'])

                # Append results for future concatenation
                datasets.append(data)

            # Concatenate results along time axis (default)
            spec = xrio.concat(datasets)

            # Update n_int after merging batches
            meta.n_int = len(spec.time)

            # Plot light curve and centroids over time
            if meta.photometry:
                if meta.isplots_S3 >= 1:
                    plots_s3.phot_lc(spec, meta)
                    plots_s3.phot_centroid(spec, meta)
                if meta.isplots_S3 >= 3:
                    plots_s3.phot_bg(spec, meta)
                if meta.isplots_S3 >= 5:
                    plots_s3.phot_npix(spec, meta)
                    plots_s3.phot_2d_frame_diff(spec, meta)
                apphot.apphot_status(spec)
                del (spec['flux'])

            # Plot fitted 2D drift
            # Note: This needs to happen before calling conclusion_step()
            if meta.isplots_S3 >= 1 and meta.inst == 'wfc3':
                plots_s3.drift_2D(spec, meta)

            if meta.inst == 'wfc3':
                # WFC3 needs a conclusion step to convert lists into
                # arrays before saving
                spec, meta, log = inst.conclusion_step(spec, meta, log)

            # make citations for current stage
            util.make_citations(meta, 3)

            # Save Dataset object containing time-series of 1D spectra
            if meta.save_output:
                meta.filename_S3_SpecData = (meta.outputdir+'S3_'+event_ap_bg +
                                             "_SpecData.h5")

                # Save Meta information to attributes of Xarray
                util.add_meta_to_xarray(meta, spec)

                success = xrio.writeXR(meta.filename_S3_SpecData, spec,
                                       verbose=True)

                if not success:
                    raise OSError('Failed to write S3_SpecData.')

            # Compute MAD value
            if not meta.photometry:
                meta.mad_s3 = util.get_mad(meta, log, spec.wave_1d.values,
                                           spec.optspec.values,
                                           spec.optmask.values,
                                           scandir=getattr(spec, 'scandir',
                                                           None))
            else:
                normspec = util.normalize_spectrum(
                    meta, spec.aplev.values,
                    scandir=getattr(spec, 'scandir', None))
                meta.mad_s3 = util.get_mad_1d(normspec)
            try:
                log.writelog(f"Stage 3 MAD = {int(np.round(meta.mad_s3))} ppm")
            except:
                log.writelog("Could not compute Stage 3 MAD")
                meta.mad_s3 = 0

            if meta.isplots_S3 >= 1 and not meta.photometry:
                log.writelog('Generating figures')
                # 2D light curve without drift correction
                plots_s3.lc_nodriftcorr(meta, spec.wave_1d, spec.optspec,
                                        optmask=spec.optmask,
                                        scandir=getattr(spec, 'scandir',
                                                        None))

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return spec, meta
