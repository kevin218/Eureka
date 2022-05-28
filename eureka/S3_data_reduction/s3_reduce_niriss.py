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

import time as time_pkg
import numpy as np
import astraeus.xarrayIO as xrio
from astropy.io import fits
from tqdm import tqdm

from . import niriss_extraction
from . import plots_s3, source_pos
from . import background as bg
from . import bright2flux as b2f
from . import niriss as inst

from ..lib import logedit
from ..lib import readECF
from ..lib import manageevent as me
from ..lib import util
from ..lib import tracing_niriss


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
    - May 2022 Adina Feinstein
        Modified and updated to work for NIRISS
    '''

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel
    meta.inst = 'niriss'

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


    # Open new log file
    meta.s3_logname = meta.outputdir + 'S3_{0}'.format(eventlabel) + ".log"
    if s2_meta is not None:
        log = logedit.Logedit(meta.s3_logname, read=s2_meta.s2_logname)
    else:
        log = logedit.Logedit(meta.s3_logname)

    # Create list of file segments
    meta = util.readfiles(meta)

    # Get summed frame for tracing
    with fits.open(meta.segment_list[-1]) as hdulist:
        # Figure out which instrument we are using
        meta.median = np.nansum(hdulist[1].data, axis=0)

    # identifies the trace for all orders
    if meta.trace_method == 'ears':
        traces = tracing_niriss.mask_method_ears(meta,
                                                 degree=meta.poly_order,
                                                 save=meta.save_table,
                                                 outdir=meta.outputdir,
                                                 isplots=meta.isplots_S3)
        meta.tab1 = traces
    elif meta.trace_method == 'edges':
        traces = tracing_niriss.mask_method_edges(meta,
                                                  radius=meta.radius,
                                                  gf=meta.filter,
                                                  save=meta.save_table,
                                                  outdir=meta.outputdir,
                                                  isplots=meta.isplots_S3)
        meta.tab2 = traces
    else:
        # This will break if traces cannot be extracted
        log.writelog('Method for identifying NIRISS trace'
                     'not implemented. Please select between "ears"'
                     'and "edges".\n')
        raise AssertionError('Method for identifying NIRISS trace'
                             'not implemented.')


    # creates mask for the traces

    # creates mask for the background pixels


    # create directories to store data
    # run_s3 used to make sure we're always looking at the right run for
    # each order

    meta.eventlabel = eventlabel

    meta.run_s3 = None
    meta.run_s3 = util.makedirectory(meta, 'S3', meta.run_s3)

    # begin process
    t0 = time_pkg.time()

    # Sets the output directory
    meta.outputdir = util.pathdirectory(meta, 'S3',
                                        meta.run_s3)

    log.writelog("\nStarting Stage 3 Reduction\n")
    log.writelog(f"Input directory: {meta.inputdir}")
    log.writelog(f"Output directory: {meta.outputdir}")

    # Copy ecf
    log.writelog('Copying S3 control file', mute=(not meta.verbose))
    meta.copy_ecf()

    # Identifies the number of files to reduce
    meta.num_data_files = len(meta.segment_list)
    if meta.num_data_files == 0:
        log.writelog(f'Unable to find any "{meta.suffix}.fits" files '
                     f'in the inputdir: \n"{meta.inputdir}"!',
                     mute=True)
        raise AssertionError(f'Unable to find any "{meta.suffix}.fits"'
                             f' files in the inputdir: \n'
                             f'"{meta.inputdir}"!')
    else:
        log.writelog(f'\nFound {meta.num_data_files} data file(s) '
                     f'ending in {meta.suffix}.fits',
                     mute=(not meta.verbose))


    datasets = []
    # Loop over each segment
    # Only reduce the last segment/file if testing_S3 is set to
    # True in ecf
    if meta.testing_S3:
        istart = meta.num_data_files - 1
    else:
        istart = 0

    for m in range(istart, meta.num_data_files):
        # Initialize data object
        data = xrio.makeDataset()

        # Report progress
        if meta.verbose:
            log.writelog(f'Reading file {m + 1} of '
                         f'{meta.num_data_files}')
        else:
            log.writelog(f'Reading file {m + 1} of '
                         f'{meta.num_data_files}', end='\r')

        # Read in data frame and header
        data, meta = inst.read(meta.segment_list[m], data, meta,
                               meta.f277_filename)

        # Get number of integrations and frame dimensions
        meta.n_int, meta.ny, meta.nx = data.flux.shape

        if meta.testing_S3:
            # Only process the last 5 integrations when testing
            meta.int_start = np.max((0, meta.n_int-5))
        else:
            meta.int_start = 0


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

        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future
        # to flag certain pixels
        data['mask'] = (['time', 'y', 'x'], np.ones(data.flux.shape,
                                                    dtype=bool))

        # Check if arrays have NaNs
        data['mask'] = util.check_nans(data['flux'], data['mask'],
                                       log, name='FLUX')
        data['mask'] = util.check_nans(data['err'], data['mask'],
                                       log, name='ERR')
        data['mask'] = util.check_nans(data['v0'], data['mask'],
                                       log, name='V0')

        # Manually mask regions [colstart, colend, rowstart, rowend]
        if hasattr(meta, 'manmask'):
            log.writelog("  Masking manually identified bad pixels",
                         mute=(not meta.verbose))
            for i in range(len(meta.manmask)):
                colstart, colend, rowstart, rowend = meta.manmask[i]
                data['mask'][rowstart:rowend, colstart:colend] = 0

        # Perform outlier rejection of sky background along time axis
        log.writelog('  Performing background outlier rejection',
                     mute=(not meta.verbose))

        data = inst.fit_bg(data, meta, log,
                           readnoise=meta.readnoise,
                           sigclip=meta.sigclip,
                           box=meta.box,
                           filter_size=meta.filter_size,
                           bkg_estimator=meta.bkg_estimator,
                           isplots=meta.isplots_S3)

        if meta.isplots_S3 >= 3:
            log.writelog('  Creating figures for background '
                         'subtraction', mute=(not meta.verbose))
            iterfn = range(meta.int_start, meta.n_int)
            if meta.verbose:
                iterfn = tqdm(iterfn)
            for n in iterfn:
                # make image+background plots
                plots_s3.image_and_background(data, meta, n, m)

        # Calulate and correct for 2D drift
        if hasattr(inst, 'correct_drift2D'):
            log.writelog('  Correcting for 2D drift',
                         mute=(not meta.verbose))
            inst.correct_drift2D(data, meta, m)

        # Select only aperture region
        ap_y1 = int(meta.src_ypos-spec_hw_val)
        ap_y2 = int(meta.src_ypos+spec_hw_val)
        apdata = data.flux[:, ap_y1:ap_y2].values
        aperr = data.err[:, ap_y1:ap_y2].values
        apmask = data.mask[:, ap_y1:ap_y2].values
        apbg = data.bg[:, ap_y1:ap_y2].values
        apv0 = data.v0[:, ap_y1:ap_y2].values
        # Compute median frame
        medapdata = np.median(apdata, axis=0)

        # Extract standard spectrum and its variance
        data['stdspec'] = (['time', 'x'], np.sum(apdata, axis=1))
        data['stdvar'] = (['time', 'x'], np.sum(aperr ** 2, axis=1))
        data['stdspec'].attrs['flux_units'] = \
            data.flux.attrs['flux_units']
        data['stdspec'].attrs['time_units'] = \
            data.flux.attrs['time_units']
        data['stdvar'].attrs['flux_units'] = \
            data.flux.attrs['flux_units']
        data['stdvar'].attrs['time_units'] = \
            data.flux.attrs['time_units']
        # FINDME: stdvar >> stdspec, which is a problem

        # Extract optimal spectrum with uncertainties
        log.writelog("  Performing optimal spectral extraction",
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

        # Already converted DN to electrons, so gain = 1 for optspex
        gain = 1
        intstart = data.attrs['intstart']
        iterfn = range(meta.int_start, meta.n_int)
        if meta.verbose:
            iterfn = tqdm(iterfn)
        for n in iterfn:
            data['optspec'][n], data['opterr'][n], mask = \
                optspex.optimize(meta, apdata[n], apmask[n], apbg[n],
                                 data.stdspec[n].values, gain, apv0[n],
                                 p5thresh=meta.p5thresh,
                                 p7thresh=meta.p7thresh,
                                 fittype=meta.fittype,
                                 window_len=meta.window_len,
                                 deg=meta.prof_deg, n=intstart+n,
                                 meddata=medapdata)

        # Mask out NaNs and Infs
        optspec_ma = np.ma.masked_invalid(data.optspec.values)
        opterr_ma = np.ma.masked_invalid(data.opterr.values)
        optmask = np.logical_or(np.ma.getmaskarray(optspec_ma),
                                np.ma.getmaskarray(opterr_ma))
        data['optmask'] = (['time', 'x'], optmask)
        # data['optspec'] = np.ma.masked_where(mask, data.optspec)
        # data['opterr'] = np.ma.masked_where(mask, data.opterr)

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

        if meta.save_output:
            # Save flux data from current segment
            filename_xr = (meta.outputdir+'S3_'+event_ap_bg +
                           "_FluxData_seg"+str(m+1).zfill(4)+".h5")
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
            data['bg'], data['mask'], data.attrs['intstart'],
            data.attrs['intend'])

        # Append results for future concatenation
        datasets.append(data)


    # Concatenate results along time axis (default)
    spec = xrio.concat(datasets)

    # Calculate total time
    total = (time_pkg.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # Save Dataset object containing time-series of 1D spectra
    meta.filename_S3_SpecData = (meta.outputdir+'S3_'+event_ap_bg +
                                 "_SpecData.h5")
    success = xrio.writeXR(meta.filename_S3_SpecData, spec,
                           verbose=True)

    # Compute MAD value
    meta.mad_s3 = util.get_mad(meta, spec.wave_1d, spec.optspec)
    log.writelog(f"Stage 3 MAD = "
                 f"{np.round(meta.mad_s3, 2).astype(int)} ppm")

    if meta.isplots_S3 >= 1:
        log.writelog('Generating figure')
        # 2D light curve without drift correction
        plots_s3.lc_nodriftcorr(meta, spec.wave_1d, spec.optspec)

    # Save results
    if meta.save_output:
        log.writelog('Saving Metadata')
        fname = meta.outputdir + 'S3_' + event_ap_bg + "_Meta_Save"
        me.saveevent(meta, fname, save=[])

    log.closelog()

    return spec, meta