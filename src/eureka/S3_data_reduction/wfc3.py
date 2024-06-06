
# WFC3 specific rountines go here
import os
import numpy as np
import multiprocessing as mp
from astropy.io import fits
import scipy.interpolate as spi
import scipy.ndimage as spni
import astraeus.xarrayIO as xrio
from . import sigrej, source_pos, background
from . import hst_scan as hst
from . import bright2flux as b2f
from ..lib import suntimecorr, utc_tt, util


def preparation_step(meta, log):
    """Perform preperatory steps which require many frames.

    Separate imaging and spectroscopy, separate observations into different
    scan directions, and calculate centroid for each frame.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.
    """
    meta.gain = 1

    meta, log = separate_direct(meta, log)
    meta, log = separate_scan_direction(meta, log)

    # Calculate centroid of direct image(s)
    # meta.centroid order is (y,x)
    meta.centroid = hst.imageCentroid(meta.direct_list, meta.centroidguess,
                                      meta.centroidtrim, meta.ny, meta.CRPIX1,
                                      meta.CRPIX2, meta.postarg1,
                                      meta.postarg2, meta, log)

    # Initialize list to hold centroid positions from later steps in this stage
    meta.centroids = []
    meta.guess = []
    meta.subdata_ref = []
    meta.subdiffmask_ref = []

    meta, log = get_reference_frames(meta, log)

    # Set to False so that Eureka! knows not to do photometry
    meta.photometry = False

    return meta, log


def get_reference_frames(meta, log):
    """Process the reference frames for each scan direction and save them
    in the meta object.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    # Temporarily override some values
    verbose = meta.verbose
    meta.verbose = False
    ncpu = meta.ncpu
    meta.ncpu = 1
    isplots_S3 = meta.isplots_S3
    meta.isplots_S3 = 0

    # Set some default values
    meta.firstFile = False
    meta.firstInBatch = False
    meta.int_start = 0
    meta.int_end = 0
    meta.files_per_batch = 1

    # Use the first two files by default
    if not hasattr(meta, 'iref'):
        raise AttributeError(
            'You must set the meta.iref parameter in your ECF for WFC3 '
            'observations. The recommended setting is [2, 3].'
        )

    # Make sure that the scan directions are in the right order
    if meta.iref[0] % 2 != 0:
        meta.iref = meta.iref[::-1]

    # Save the reference frame for each scan direction
    for i in meta.iref:
        log.writelog(f"Capturing info from reference frame {i}...")
        data = xrio.makeDataset()
        data, meta, log = read(meta.segment_list[i], data, meta, log)
        meta.n_int, meta.ny, meta.nx = data.flux.shape
        data, meta = util.trim(data, meta)
        # Create bad pixel mask (1 = good, 0 = bad)
        data['mask'] = (['time', 'y', 'x'],
                        np.ones(data.flux.shape, dtype=bool))
        data['mask'] = util.check_nans(data['flux'], data['mask'],
                                       log, name='FLUX')
        data['mask'] = util.check_nans(data['err'], data['mask'],
                                       log, name='ERR')
        data['mask'] = util.check_nans(data['v0'], data['mask'],
                                       log, name='V0')
        if hasattr(meta, 'manmask'):
            util.manmask(data, meta, log)
        # Need to add guess after trimming and before cut_aperture
        meta.guess.append(data.guess)
        data, meta, log = source_pos.source_pos_wrapper(data, meta, log, i)
        data, meta = b2f.convert_to_e(data, meta, log)
        data = flag_bg(data, meta, log)
        data = background.BGsubtraction(data, meta, log, i)
        cut_aperture(data, meta, log)

        # Save the reference values
        meta.subdata_ref.append(data.flux)
        meta.subdiffmask_ref.append(data.flatmask)

    # Restore input values
    meta.verbose = verbose
    meta.ncpu = ncpu
    meta.isplots_S3 = isplots_S3

    return meta, log


def conclusion_step(data, meta, log):
    """Convert lists into arrays for saving and applies meta.sum_reads
    if requested.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.
    """
    meta.centroids = np.array(meta.centroids)
    meta.guess = np.array(meta.guess)
    meta.subdata_ref = np.array(meta.subdata_ref)
    meta.subdiffmask_ref = np.array(meta.subdiffmask_ref)

    # Delete the no-longer needed scandir attribute
    delattr(meta, 'scandir')

    return data, meta, log


def separate_direct(meta, log):
    """Separate out the direct observations from the science observations.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    obstimes : ndarray
        The times of each integration.
    CRPIX1 : float
        The CRPIX1 FITS header value.
    CRPIX2 : float
        The CRPIX2 FITS header value.
    postarg1 : float
        The POSTARG1 FITS header value.
    postarg2 : float
        The POSTARG2 FITS header value.
    ny : int
        The NAXIS2 FITS header value.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.

    Raises
    ------
    AssertionError
        All observations cannot be in imaging mode.
    AssertionError
        All observations cannot be spectroscopic.
    AssertionError
        Unknown OBSTYPE(s) encountered.
    """
    # Figure out which files are IMAGING or SPECTROSCOPIC
    obstypes = []
    obstimes = []
    postarg1 = []
    postarg2 = []
    CRPIX1 = []
    CRPIX2 = []
    for fname in meta.segment_list:
        with fits.open(fname) as file:
            obstypes.append(file[0].header['OBSTYPE'])
            obstimes.append(file[0].header['EXPSTART'])
            # Get the POSTARG2 parameter so we can
            # later separate scan directions
            postarg1.append(file[0].header['POSTARG1'])
            postarg2.append(file[0].header['POSTARG2'])
            CRPIX1.append(file[1].header['CRPIX1'])
            CRPIX2.append(file[1].header['CRPIX2'])
            ny = file[1].header['NAXIS2']
    obstypes = np.array(obstypes)
    obstimes = np.array(obstimes)
    postarg1 = np.array(postarg1)
    postarg2 = np.array(postarg2)
    CRPIX1 = np.array(CRPIX1)
    CRPIX2 = np.array(CRPIX2)

    # Make sure all the files are in order of observation time
    order = np.argsort(obstimes)
    meta.segment_list = meta.segment_list[order]
    obstypes = obstypes[order]
    obstimes = obstimes[order]
    postarg1 = postarg1[order]
    postarg2 = postarg2[order]
    CRPIX1 = CRPIX1[order]
    CRPIX2 = CRPIX2[order]

    if np.all(obstypes == 'IMAGING'):
        # All observations are in imaging mode
        raise AssertionError('All observations cannot be in imaging mode!\n'
                             'Eureka is currently not capable of handling '
                             'imaging datasets from Hubble/WFC3.')
    elif np.all(obstypes == 'SPECTROSCOPIC'):
        # All observations are in spectroscopy mode
        # This is an issue as an imaging mode observation is needed
        # for wavelength calibration
        raise AssertionError('All observations cannot be spectroscopic!\n'
                             'At least one direct image is needed for '
                             'wavelength calibration.')
    elif np.any(np.logical_and(obstypes != 'SPECTROSCOPIC',
                               obstypes != 'IMAGING')):
        # There is one or more unexpected OBSTYPEs - throw a useful error
        unknowns = obstypes[np.logical_and(obstypes != 'SPECTROSCOPIC',
                                           obstypes != 'IMAGING')]
        unknowns = np.unique(unknowns)
        raise AssertionError(f'Unknown OBSTYPE(s) encountered: {unknowns}.\n'
                             'Expected only SPECTROSCOPIC and IMAGING '
                             'OBSTYPEs.')
    else:
        # There is a mix of some direct images for wavelength calibration
        # and science spectra as expected

        # Make separate lists of direct images and science images
        meta.direct_list = meta.segment_list[obstypes == 'IMAGING']
        meta.n_img = len(meta.direct_list)
        meta.segment_list = meta.segment_list[obstypes == 'SPECTROSCOPIC']
        meta.num_data_files = len(meta.segment_list)
        postarg1 = postarg1[obstypes == 'SPECTROSCOPIC']
        postarg2 = postarg2[obstypes == 'SPECTROSCOPIC']
        CRPIX1 = CRPIX1[obstypes == 'SPECTROSCOPIC'][0]
        CRPIX2 = CRPIX2[obstypes == 'SPECTROSCOPIC'][0]

        # Figure out which direct image should be used by each science image
        # If there are multiple, this will use the most recent one
        # If there haven't been any yet, will use the next one
        direct_times = obstimes[obstypes == 'IMAGING']
        science_times = obstimes[obstypes == 'SPECTROSCOPIC']
        meta.direct_index = np.zeros(meta.segment_list.shape, dtype=int)
        for i in range(len(science_times)):
            indices = np.where(science_times[i] > direct_times)[0]
            if len(indices) == 0:
                index = 0
            else:
                index = indices[-1]
            meta.direct_index[i] = index

    meta.obstimes = science_times
    meta.CRPIX1 = CRPIX1
    meta.CRPIX2 = CRPIX2
    meta.postarg1 = postarg1
    meta.postarg2 = postarg2
    meta.ny = ny

    return meta, log


def separate_scan_direction(meta, log):
    """Separate alternating scan directions.

    Parameters
    ----------
    obstimes : ndarray
        The times for each integration.
    postarg2 : float
        The POSTARG2 FITS header value.
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.
    """
    if meta.num_data_files == 1:
        # There is only one image
        meta.scandir = np.zeros(meta.num_data_files, dtype=int)
        meta.n_scan0 = 1
        meta.n_scan1 = 0
    else:
        # Assign scan direction
        meta.scandir = np.zeros(meta.num_data_files, dtype=int)
        meta.n_scan0 = 0
        meta.n_scan1 = 0
        scan0 = meta.postarg2[0]
        scan1 = meta.postarg2[1]
        for m in range(meta.num_data_files):
            if meta.postarg2[m] == scan0:
                meta.n_scan0 += 1
            elif meta.postarg2[m] == scan1:
                meta.scandir[m] = 1
                meta.n_scan1 += 1
            else:
                log.writelog(f'WARNING: Unknown scan direction for file {m}.')

    log.writelog(f"# of files in scan direction 0: {meta.n_scan0}",
                 mute=(not meta.verbose))
    log.writelog(f"# of files in scan direction 1: {meta.n_scan1}",
                 mute=(not meta.verbose))

    # Group frames into frame, batch, and orbit number
    meta.framenum, meta.batchnum, meta.orbitnum = \
        hst.groupFrames(meta.obstimes)

    return meta, log


def read(filename, data, meta, log):
    '''Reads single FITS file from HST's WFC3 instrument.

    Parameters
    ----------
    filename : str
        Single filename to read
    data : Xarray Dataset
        The Dataset object in which the fits data will stored
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The current log.

    Notes
    -----
    History:

    - January 2017 Kevin Stevenson
        Initial code as implemented in the WFC3 pipeline
    - 18-19 Nov 2021 Taylor Bell
        Edited and decomposed WFC3 code to integrate with Eureka!
    - May 9, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''
    # Determine image size and filter/grism
    with fits.open(filename) as hdulist:
        data.attrs['filename'] = filename
        data.attrs['mhdr'] = hdulist[0].header
        data.attrs['shdr'] = hdulist[1].header
        meta.nx = data.attrs['shdr']['NAXIS1']
        meta.ny = data.attrs['shdr']['NAXIS2']
        meta.grism = data.attrs['mhdr']['FILTER']
        meta.detector = data.attrs['mhdr']['DETECTOR']
        meta.flatoffset = [[-1*data.attrs['shdr']['LTV2'],
                            -1*data.attrs['shdr']['LTV1']]]
        data.attrs['exptime'] = data.attrs['mhdr']['EXPTIME']
        flux_units = data.attrs['shdr']['BUNIT']

        # Determine if we are using IMA or FLT files
        if filename.endswith('flt.fits'):
            # FLT files subtract first from last, 2 reads
            meta.nreads = 1
        else:
            meta.nreads = data.attrs['shdr']['SAMPNUM']

        sci = np.zeros((meta.nreads, meta.ny, meta.nx))  # Flux
        err = np.zeros((meta.nreads, meta.ny, meta.nx))  # Error
        dq = np.zeros((meta.nreads, meta.ny, meta.nx))  # Flags
        jd = []
        for j, rd in enumerate(range(meta.nreads, 0, -1)):
            sci[j] = hdulist['SCI', rd].data
            err[j] = hdulist['ERR', rd].data
            dq[j] = hdulist['DQ', rd].data
            jd.append(2400000.5+hdulist['SCI', rd].header['ROUTTIME']
                      - 0.5*hdulist['SCI', rd].header['DELTATIM']/3600/24)
        jd = np.array(jd)

    ra = data.attrs['mhdr']['RA_TARG']*np.pi/180
    dec = data.attrs['mhdr']['DEC_TARG']*np.pi/180
    frametime = (2400000.5+0.5*(data.attrs['mhdr']['EXPSTART']
                                + data.attrs['mhdr']['EXPEND']))
    if meta.horizonsfile is not None:
        horizon_path = os.path.join(meta.hst_cal,
                                    *meta.horizonsfile.split(os.sep))
    if meta.horizonsfile is not None and os.path.isfile(horizon_path):
        # Apply light-time correction, convert to BJD_TDB
        # Horizons file created for HST around time of observations
        bjd_corr = suntimecorr.suntimecorr(ra, dec, jd, horizon_path)
        bjdutc = jd + bjd_corr/86400.
        # FINDME: this was utc_tt, but I believe it should have
        # been utc_tdb instead
        if not hasattr(meta, 'leapdir') or meta.leapdir is None:
            meta.leapdir = 'leapdir'
        leapdir_path = os.path.join(meta.hst_cal,
                                    *meta.leapdir.split(os.sep))
        if leapdir_path[-1] != os.sep:
            leapdir_path += os.sep
        time = utc_tt.utc_tdb(bjdutc, leapdir_path, log)
        frametime = utc_tt.utc_tdb(frametime+bjd_corr/86400., leapdir_path,
                                   log)
        time_units = 'BJD_TDB'
    else:
        if meta.firstFile:
            print("WARNING: No Horizons file found. Using JD rather than "
                  "BJD_TDB.")
        time = jd
        time_units = 'HJD_UTC'
    data.attrs['frametime'] = frametime

    # Create flux-like DataArrays
    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')

    # Calculate centroids for each frame
    centroids = np.zeros((meta.nreads, 2))
    # Figure out which direct image is the relevant one for this observation
    image_number = np.where(meta.segment_list == filename)[0][0]
    centroid_index = meta.direct_index[image_number]
    # Use the same centroid for each read
    centroids[:, 0] = meta.centroid[centroid_index][0]
    centroids[:, 1] = meta.centroid[centroid_index][1]
    meta.centroids.append(centroids)

    # Calculate trace
    if meta.firstInBatch:
        log.writelog(f"  Calculating wavelength assuming {meta.grism} "
                     f"filter/grism...", mute=(not meta.verbose))
    xrange = np.arange(0, meta.nx)
    # wavelength in microns
    wave = hst.calibrateLambda(xrange, centroids[0], meta.grism)/1e4
    # Assume no skew over the detector
    wave_2d = wave*np.ones((meta.ny, 1))
    wave_units = 'microns'
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    # Divide data by flat field
    if meta.flatfile is None and meta.firstFile:
        log.writelog('No flat frames found.')
    else:
        data, meta, log = flatfield(data, meta, log)

    # Compute differences between non-destructive reads
    diffdata, meta, log = difference_frames(data, meta, log)

    # Determine read noise and gain
    readNoise = np.mean((data.attrs['mhdr']['READNSEA'],
                         data.attrs['mhdr']['READNSEB'],
                         data.attrs['mhdr']['READNSEC'],
                         data.attrs['mhdr']['READNSED']))
    v0 = readNoise**2*np.ones_like(diffdata.flux.values)  # Units of electrons
    diffdata['v0'] = (['time', 'y', 'x'], v0)

    # Assign dq to diffdata
    # This is a bit of a hack, but dq is not currently being used
    diffdata['dq'] = data.dq[:-1]

    # Assign wavelength to diffdata
    diffdata['wave'] = (['x'], wave)
    diffdata['wave'].attrs['wave_units'] = wave_units
    diffdata['wave_2d'] = (['y', 'x'], wave_2d)
    diffdata['wave_2d'].attrs['wave_units'] = wave_units

    # Figure out which read this file starts and ends with
    diffdata.attrs['intstart'] = image_number*(meta.nreads-1)
    diffdata.attrs['intend'] = (image_number+1)*(meta.nreads-1)

    # Copy science and master headers
    diffdata.attrs['shdr'] = data.attrs['shdr']
    diffdata.attrs['mhdr'] = data.attrs['mhdr']
    diffdata.attrs['filename'] = data.attrs['filename']

    diffdata['scandir'] = (['time'], np.repeat(meta.scandir[filename ==
                                                            meta.segment_list],
                                               meta.nreads))

    return diffdata, meta, log


def flatfield(data, meta, log):
    '''Perform flatfielding.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with flatfielding applied.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    '''
    if meta.firstInBatch:
        log.writelog(f'  Performing flat fielding using:\n'
                     f'    {meta.flatfile}',
                     mute=(not meta.verbose))
    flatfile_path = os.path.join(meta.hst_cal,
                                 *meta.flatfile.split(os.sep))
    # Make list of master flat field frames
    tempflat, tempmask = hst.makeflats(flatfile_path,
                                       [np.mean(data.wave_2d.values,
                                                axis=0), ],
                                       [[0, meta.nx], ], [[0, meta.ny], ],
                                       meta.flatoffset, 1, meta.ny, meta.nx,
                                       sigma=meta.flatsigma,
                                       isplots=meta.isplots_S3)
    subflat = tempflat[0]
    flatmask = tempmask[0]

    time_units = data.flux.attrs['time_units']
    data['flatmask'] = xrio.makeFluxLikeDA(flatmask[np.newaxis],
                                           data.time.values[:1], "None",
                                           time_units, name='flatmask')

    # Calculate reduced image
    subflat[np.where(flatmask == 0)] = 1
    subflat[np.where(subflat == 0)] = 1
    data['flux'] /= subflat

    return data, meta, log


def difference_frames(data, meta, log):
    '''Compute differenced frames.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object differenced frames.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    '''
    if meta.nreads > 1 and meta.firstInBatch:
        log.writelog('  Differencing non-destructive reads...',
                     mute=(not meta.verbose))

    if meta.nreads > 1:
        # Subtract pairs of subframes
        meta.nreads -= 1
        diffflux = np.zeros((meta.nreads, meta.ny, meta.nx))
        differr = np.zeros((meta.nreads, meta.ny, meta.nx))
        for n in range(meta.nreads):
            diffflux[n] = data.flux[n+1]-data.flux[n]
            differr[n-1] = np.sqrt(data.err[n]**2+data.err[n-1]**2)
    else:
        # FLT data has already been differenced
        diffflux = data.flux
        differr = data.err

    # Temporarily set this value for now
    meta.n_int = meta.nreads

    diffmask = np.zeros((meta.nreads, meta.ny, meta.nx))
    guess = np.zeros((meta.nreads), dtype=int)
    for n in range(meta.nreads):
        diffmask[n] = data['flatmask'][0][0]
        if meta.nreads > 1:
            diffmask[n][np.where(differr[n] > meta.diffthresh *
                        np.median(differr[n], axis=1)[:, np.newaxis])] = 0
        else:
            # Don't use diffthresh for FLT files
            pass

        # Guess spectrum position only using subarray region
        masked_data = diffflux[n, meta.ywindow[0]:meta.ywindow[1],
                               meta.xwindow[0]:meta.xwindow[1]] * \
            diffmask[n, meta.ywindow[0]:meta.ywindow[1],
                     meta.xwindow[0]:meta.xwindow[1]]
        guess[n] = (np.median(np.where(masked_data > np.mean(masked_data))[0])
                    + meta.ywindow[0]).astype(int)
    # Guess may be skewed if first read is zeros
    if guess[0] < 0 or guess[0] > meta.ny:
        guess[0] = guess[1]

    # Compute full scan length
    if meta.firstInBatch:
        log.writelog('  Computing scan height...',
                     mute=(not meta.verbose))
    scanHeight = []
    for i in range(meta.n_int):
        scannedData = np.sum(data.flux[i], axis=1)
        xmin = np.min(guess)
        xmax = np.max(guess)
        scannedData /= np.median(scannedData[xmin:xmax+1])
        scannedData -= 0.5
        yrng = range(meta.ny)
        spline = spi.UnivariateSpline(yrng, scannedData[yrng], k=3, s=0)
        roots = spline.roots()
        scanHeight.append(roots[1]-roots[0])

    # Create Xarray Dataset with updated time axis for differenced frames
    flux_units = data.flux.attrs['flux_units']
    time_units = data.flux.attrs['time_units']
    if meta.nreads > 1:
        difftime = data.time[:-1] + 0.5*np.ediff1d(data.time)
    else:
        # FLT data has already been differenced
        difftime = data.time
    diffdata = xrio.makeDataset()
    diffdata['flux'] = xrio.makeFluxLikeDA(diffflux, difftime, flux_units,
                                           time_units, name='flux')
    diffdata['err'] = xrio.makeFluxLikeDA(differr, difftime, flux_units,
                                          time_units, name='err')
    diffdata['flatmask'] = xrio.makeFluxLikeDA(diffmask, difftime, "None",
                                               time_units, name='mask')
    variance = np.zeros_like(diffdata.flux.values)
    diffdata['variance'] = xrio.makeFluxLikeDA(variance, difftime, flux_units,
                                               time_units, name='variance')
    diffdata['guess'] = xrio.makeTimeLikeDA(guess, difftime, 'pixels',
                                            time_units, 'guess')
    diffdata['scanHeight'] = xrio.makeTimeLikeDA(scanHeight, difftime,
                                                 'pixels', time_units,
                                                 'scanHeight')

    return diffdata, meta, log


def flag_bg(data, meta, log):
    '''Outlier rejection of sky background along time axis.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with outlier background pixels flagged.
    '''
    log.writelog('  Performing background outlier rejection...',
                 mute=(not meta.verbose))

    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        if len(iscans) > 0:
            for n in range(meta.nreads):
                iscan = iscans[n::meta.nreads]
                # Set limits on the sky background
                x1 = (data.guess.values[iscan].min()-meta.bg_hw).astype(int)
                x2 = (data.guess.values[iscan].max()+meta.bg_hw).astype(int)
                bgdata1 = data.flux[iscan, :x1]
                bgmask1 = data.flux[iscan, :x1]
                bgdata2 = data.flux[iscan, x2:]
                bgmask2 = data.flux[iscan, x2:]
                if hasattr(meta, 'use_estsig') and meta.use_estsig:
                    bgerr1 = np.median(data.err[iscan, :x1])
                    bgerr2 = np.median(data.err[iscan, x2:])
                    estsig1 = [bgerr1 for j in range(len(meta.bg_thresh))]
                    estsig2 = [bgerr2 for j in range(len(meta.bg_thresh))]
                else:
                    estsig1 = None
                    estsig2 = None
                data['mask'][iscan, :x1] = sigrej.sigrej(bgdata1,
                                                         meta.bg_thresh,
                                                         bgmask1, estsig1)
                data['mask'][iscan, x2:] = sigrej.sigrej(bgdata2,
                                                         meta.bg_thresh,
                                                         bgmask2, estsig2)

    return data


def fit_bg(dataim, datamask, datav0, datavariance, guess, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam, but adds on some extra steps

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        An array of which data should be masked.
    datav0 : ndarray (2D)
        readNoise**2.
    datavariance : ndarray (2D)
        Initially an all zeros array.
    n : int
        The current integration.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
        The plotting verbosity, by default False.

    Returns
    -------
    bg : ndarray (2D)
        The fitted background level.
    mask : ndarray (2D)
        The updated mask after background subtraction.
    datav0 : ndarray (2D)
        readNoise**2+np.mean(bgerr**2)
    datavariance : ndarray (2D)
        abs(dataim) / meta.gain + datav0
    n : int
        The current integration number.
    """
    y2 = guess + meta.bg_hw
    y1 = guess - meta.bg_hw

    bg, mask = background.fitbg(dataim, meta, datamask, y1, y2,
                                deg=meta.bg_deg, threshold=meta.p3thresh,
                                isrotate=2, isplots=isplots)

    # Calculate variance assuming background dominated rather than
    # read noise dominated
    bgerr = np.std(bg, axis=0)/np.sqrt(np.sum(datamask, axis=0))
    bgerr[np.logical_not(np.isfinite(bgerr))] = 0.
    datav0 += np.mean(bgerr**2)
    datavariance = abs(dataim) / meta.gain + datav0

    return bg, mask, datav0, datavariance, n


def correct_drift2D(data, meta, log, m):
    """Correct for calculated 2D drift.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The current file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object after 2D drift correction.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The current log.
    """
    def writeDrift2D(arg):
        value, n = arg
        # Assign to array of spectra and uncertainties
        drift2D[n] = value
        return

    log.writelog("  Calculating 2D drift...", mute=(not meta.verbose))
    drift2D = np.zeros((meta.n_int, 2))
    if meta.ncpu == 1:
        # Only 1 CPU
        for n in range(meta.n_int):
            # Get read number
            r = n % meta.nreads
            # Get index of reference frame
            # (0 = forward scan, 1 = reverse scan)
            p = data.scandir.values[n]
            writeDrift2D(hst.calcDrift2D((meta.subdata_ref[p][r] *
                                          meta.subdiffmask_ref[p][r]),
                                         (data.flux[n]*data.flatmask[n]),
                                         n))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        for n in range(meta.n_int):
            # Get read number
            r = n % meta.nreads
            # Get index of reference frame
            # (0 = forward scan, 1 = reverse scan)
            p = data.scandir.values[n]
            res = pool.apply_async(hst.calcDrift2D,
                                   args=((meta.subdata_ref[p][r] *
                                          meta.subdiffmask_ref[p][r]),
                                         (data.flux[n]*data.flatmask[n]),
                                         n),
                                   callback=writeDrift2D)
        pool.close()
        pool.join()
        res.wait()

    # Save the fitted drifts in the data object
    data['centroid_x'] = (['time'], drift2D[:, 0])
    data.centroid_x.attrs['units'] = 'pixels'
    data['centroid_y'] = (['time'], drift2D[:, 1])
    data.centroid_y.attrs['units'] = 'pixels'

    log.writelog("  Performing rough, pixel-scale drift correction...",
                 mute=(not meta.verbose))
    drift2D_int = np.round(drift2D, 0)
    # Correct for drift by integer pixel numbers, no interpolation
    for n in range(meta.n_int):
        data.flux[n] = spni.shift(data.flux[n],
                                  -1*drift2D_int[n, ::-1], order=0,
                                  mode='constant', cval=0)
        data.mask[n] = spni.shift(data.mask[n],
                                  -1*drift2D_int[n, ::-1], order=0,
                                  mode='constant', cval=0)
        data.variance[n] = spni.shift(data.variance[n],
                                      -1*drift2D_int[n, ::-1],
                                      order=0, mode='constant', cval=0)
        data.bg[n] = spni.shift(data.bg[n],
                                -1*drift2D_int[n, ::-1], order=0,
                                mode='constant', cval=0)

    # Outlier rejection of full frame along time axis
    if meta.files_per_batch == 1 and meta.firstFile:
        log.writelog("  WARNING: It is recommended to run Eureka! in batch\n"
                     "  mode (nfiles >> 1) for WFC3 data to allow full-frame\n"
                     "  outlier rejection.")
    elif meta.files_per_batch > 1:
        log.writelog("  Performing full-frame outlier rejection...",
                     mute=(not meta.verbose))
        for p in range(2):
            iscans = np.where(data.scandir.values == p)[0]
            if len(iscans) > 0:
                for n in range(meta.nreads):
                    iscan = iscans[n::meta.nreads]
                    data.mask[iscan] = sigrej.sigrej(data.flux[iscan],
                                                     meta.bg_thresh,
                                                     data.mask[iscan])

    log.writelog("  Performing sub-pixel drift correction...",
                 mute=(not meta.verbose))
    # Get indices for each pixel
    ix = range(meta.subnx)
    iy = range(meta.subny)
    # Define the degrees of the bivariate spline
    kx, ky = (1, 1)  # FINDME: should be using (3,3)
    # Correct for drift
    for n in range(meta.n_int):
        # Need to swap ix and iy because of numpy
        spline = spi.RectBivariateSpline(iy, ix, data.flux[n], kx=kx,
                                         ky=ky, s=0)
        # Need to subtract drift2D since documentation says (where im1 is
        # the reference image)
        # "Measures the amount im2 is offset from im1 (i.e., shift im2 by
        # -1 * these #'s to match im1)"
        data.flux[n] = spline((iy-drift2D[n, 1] +
                               drift2D_int[n, 1]).flatten(),
                              (ix-drift2D[n, 0] +
                               drift2D_int[n, 0]).flatten())
        # Need to be careful with shifting the mask. Do the shifting, and
        # mask whichever pixel was closest to the one that had been masked
        spline = spi.RectBivariateSpline(iy, ix, data.mask[n], kx=kx,
                                         ky=ky, s=0)
        data.mask[n] = spline((iy-drift2D[n, 1] +
                               drift2D_int[n, 1]).flatten(),
                              (ix-drift2D[n, 0] +
                               drift2D_int[n, 0]).flatten())
        # Fractional masking won't work - make sure it is all integer
        data.mask[n] = np.round(data.mask[n]).astype(int)
        spline = spi.RectBivariateSpline(iy, ix, data.variance[n], kx=kx,
                                         ky=ky, s=0)
        data.variance[n] = spline((iy-drift2D[n, 1] +
                                   drift2D_int[n, 1]).flatten(),
                                  (ix-drift2D[n, 0] +
                                   drift2D_int[n, 0]).flatten())
        spline = spi.RectBivariateSpline(iy, ix, data.bg[n], kx=kx,
                                         ky=ky, s=0)
        data.bg[n] = spline((iy-drift2D[n, 1] +
                             drift2D_int[n, 1]).flatten(),
                            (ix-drift2D[n, 0] +
                             drift2D_int[n, 0]).flatten())

    return data, meta, log


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    apdata : ndarray
        The flux values over the aperture region.
    aperr : ndarray
        The noise values over the aperture region.
    apmask : ndarray
        The mask values over the aperture region.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version, edited to work for HST scanned observations.
    """
    log.writelog('  Extracting aperture region...',
                 mute=(not meta.verbose))

    apdata = np.zeros((meta.n_int, meta.spec_hw*2+1, meta.subnx))
    aperr = np.zeros((meta.n_int, meta.spec_hw*2+1, meta.subnx))
    apmask = np.zeros((meta.n_int, meta.spec_hw*2+1, meta.subnx))
    apbg = np.zeros((meta.n_int, meta.spec_hw*2+1, meta.subnx))
    apv0 = np.zeros((meta.n_int, meta.spec_hw*2+1, meta.subnx))

    for f in range(int(meta.n_int/meta.nreads)):
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p = data.scandir[f*meta.nreads].values
        for r in range(meta.nreads):
            # Figure out the index currently being cut out
            n = f*meta.nreads + r

            # Use the centroid from the relevant reference frame
            guess = meta.guess[p].values[r]

            ap_y1 = (guess-meta.spec_hw).astype(int)
            ap_y2 = (guess+meta.spec_hw+1).astype(int)

            if ap_y1 < 0:
                ap_y1 = 0
                ap_y2 = 2*meta.spec_hw + 1

            if ap_y2 > len(data.flux.values[n]):
                ap_y2 = len(data.flux.values[n])
                ap_y1 = len(data.flux.values[n]) - (2*meta.spec_hw + 1)

            # Cut out this particular read
            apdata[n] = data.flux.values[n, ap_y1:ap_y2]
            aperr[n] = data.err.values[n, ap_y1:ap_y2]
            apmask[n] = data.mask.values[n, ap_y1:ap_y2]
            apbg[n] = data.bg.values[n, ap_y1:ap_y2]
            apv0[n] = data.v0.values[n, ap_y1:ap_y2]

    return apdata, aperr, apmask, apbg, apv0
