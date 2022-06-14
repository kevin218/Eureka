
# WFC3 specific rountines go here
import numpy as np
import multiprocessing as mp
from astropy.io import fits
import scipy.interpolate as spi
import scipy.ndimage as spni
import astraeus.xarrayIO as xrio
from . import nircam
from . import hst_scan as hst
from ..lib import suntimecorr, utc_tt


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
    meta.centroid = hst.imageCentroid(meta.direct_list, meta.centroidguess,
                                      meta.centroidtrim, meta.ny, meta.CRPIX1,
                                      meta.CRPIX2, meta.postarg1,
                                      meta.postarg2, meta, log)

    # Initialize list to hold centroid positions from later steps in this stage
    meta.centroids = []
    meta.subflat = []
    meta.flatmask = []
    meta.scanHeight = []
    meta.diffmask = []
    meta.subdiffmask = []
    meta.drift2D = []
    meta.drift2D_int = []
    meta.subdata_ref = []
    meta.subdiffmask_ref = []

    return meta, log


def conclusion_step(meta, log):
    """Convert lists into arrays for saving

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
    meta.centroids = np.array(meta.centroids)
    meta.subflat = np.array(meta.subflat)
    meta.flatmask = np.array(meta.flatmask)
    meta.scanHeight = np.array(meta.scanHeight)
    meta.diffmask = np.array(meta.diffmask)
    meta.subdiffmask = np.array(meta.subdiffmask)
    meta.drift2D = np.array(meta.drift2D)
    meta.drift2D_int = np.array(meta.drift2D_int)
    meta.subdata_ref = np.array(meta.subdata_ref)
    meta.subdiffmask_ref = np.array(meta.subdiffmask_ref)

    return meta, log


def separate_direct(meta, log):
    """_summary_

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
        # If there are multiple direct images, this will usethe most recent one
        direct_times = obstimes[obstypes == 'IMAGING']
        science_times = obstimes[obstypes == 'SPECTROSCOPIC']
        meta.direct_index = np.zeros(meta.segment_list.shape, dtype=int)
        for i in range(len(science_times)):
            meta.direct_index[i] = \
                np.where(science_times[i] > direct_times)[0][-1]

    meta.obstimes = obstimes
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

    # Order frames by scan direction (for batch processing)
    # order = np.append(np.where(meta.scandir == 0)[0],
    #                   np.where(meta.scandir == 1)[0])
    # meta.segment_list = meta.segment_list[order]
    # meta.scandir = meta.scandir[order]
    # meta.obstimes = meta.obstimes[order]
    # meta.postarg1 = meta.postarg1[order]
    # meta.postarg2 = meta.postarg2[order]
    # meta.direct_index = meta.direct_index[order]

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
    data : DataClass
        The updated data object with the fits data stored inside
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
            meta.nreads = 2
        else:
            meta.nreads = data.attrs['shdr']['SAMPNUM']
        
        data.attrs['nreads'] = meta.nreads

        sci = np.zeros((meta.nreads, meta.ny, meta.nx))  # Flux
        err = np.zeros((meta.nreads, meta.ny, meta.nx))  # Error
        dq = np.zeros((meta.nreads, meta.ny, meta.nx))  # Flags
        jd = []
        j = 0
        for rd in range(meta.nreads, 0, -1):
            sci[j] = hdulist['SCI', rd].data
            err[j] = hdulist['ERR', rd].data
            dq[j] = hdulist['DQ', rd].data
            jd.append(2400000.5+hdulist['SCI', rd].header['ROUTTIME']
                      - 0.5*hdulist['SCI', rd].header['DELTATIM']/3600/24)
            j += 1
        jd = np.array(jd)

    ra = data.attrs['mhdr']['RA_TARG']*np.pi/180
    dec = data.attrs['mhdr']['DEC_TARG']*np.pi/180
    frametime = (2400000.5+0.5*(data.attrs['mhdr']['EXPSTART']
                                + data.attrs['mhdr']['EXPEND']))
    if meta.horizonsfile is not None:
        # Apply light-time correction, convert to BJD_TDB
        # Horizons file created for HST around time of observations
        bjd_corr = suntimecorr.suntimecorr(ra, dec, jd, meta.horizonsfile)
        bjdutc = jd + bjd_corr/86400.
        # FINDME: this was utc_tt, but I believe it should have
        # been utc_tdb instead
        time = utc_tt.utc_tdb(bjdutc, meta.leapdir, log)
        frametime = utc_tt.utc_tdb(frametime+bjd_corr/86400., meta.leapdir,
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
    diffdata.attrs['nreads'] = data.attrs['nreads']

    return diffdata, meta, log


def flatfield(data, meta, log):
    '''Perform flatfielding.

    Parameters
    ----------
    data : DataClass
        The data object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : DataClass
        The updated data object with flatfielding applied.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    '''
    if meta.firstInBatch:
        log.writelog('  Performing flat fielding.', mute=(not meta.verbose))
    # Make list of master flat field frames
    if meta.firstFile:
        log.writelog(f'  Loading flat frame:\n  {meta.flatfile}',
                     mute=(not meta.verbose))
    tempflat, tempmask = hst.makeflats(meta.flatfile,
                                       [np.mean(data.wave_2d.values,
                                                axis=0), ],
                                       [[0, meta.nx], ], [[0, meta.ny], ],
                                       meta.flatoffset, 1, meta.ny, meta.nx,
                                       sigma=meta.flatsigma,
                                       isplots=meta.isplots_S3)
    subflat = tempflat[0]
    flatmask = tempmask[0]

    meta.subflat.append(subflat)
    meta.flatmask.append(flatmask)

    # Calculate reduced image
    subflat[np.where(flatmask == 0)] = 1
    subflat[np.where(subflat == 0)] = 1
    data['flux'] /= subflat

    return data, meta, log


def difference_frames(data, meta, log):
    '''Compute differenced frames.

    Parameters
    ----------
    data : DataClass
        The data object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : DataClass
        The updated data object with differenced frames.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    '''
    if meta.nreads > 1:
        # Subtract pairs of subframes
        meta.nreads -= 1
        data.attrs['nreads'] -= 1
        diffflux = np.zeros((meta.nreads, meta.ny, meta.nx))
        differr = np.zeros((meta.nreads, meta.ny, meta.nx))
        for n in range(meta.nreads):
            diffflux[n] = data.flux[n+1]-data.flux[n]
            differr[n-1] = np.sqrt(data.err[n]**2+data.err[n-1]**2)
    else:
        # FLT data has already been differenced
        diffflux = data.flux
        differr = data.err

    diffmask = np.zeros((meta.nreads, meta.ny, meta.nx))
    guess = np.zeros((meta.nreads), dtype=int)
    for n in range(meta.nreads):
        diffmask[n] = np.copy(meta.flatmask[-1][0])
        try:
            diffmask[n][np.where(differr[n] > meta.diffthresh *
                        np.median(differr[n], axis=1)[:, np.newaxis])] = 0
        except:
            # FINDME: Need to only catch the expected exception
            # May fail for FLT files
            log.writelog("Diffthresh failed - this may happen for FLT files.")

        masked_data = diffflux[n]*diffmask[n]
        guess[n] = np.median(np.where(masked_data > np.mean(masked_data)
                                      )[0]).astype(int)
    # Guess may be skewed if first read is zeros
    if guess[0] < 0 or guess[0] > meta.ny:
        guess[0] = guess[1]

    # Compute full scan length
    scannedData = np.sum(data.flux[-1], axis=1)
    xmin = np.min(guess)
    xmax = np.max(guess)
    scannedData /= np.median(scannedData[xmin:xmax+1])
    scannedData -= 0.5
    yrng = range(meta.ny)
    spline = spi.UnivariateSpline(yrng, scannedData[yrng], k=3, s=0)
    roots = spline.roots()
    try:
        meta.scanHeight.append(roots[1]-roots[0])
    except:
        # FINDME: Need to only catch the expected exception
        pass

    # Create Xarray Dataset with updated time axis for differenced frames
    flux_units = data.flux.attrs['flux_units']
    time_units = data.flux.attrs['time_units']
    difftime = data.time[:-1] + 0.5*np.ediff1d(data.time)
    diffdata = xrio.makeDataset()
    diffdata['flux'] = xrio.makeFluxLikeDA(diffflux, difftime, flux_units,
                                           time_units, name='flux')
    diffdata['err'] = xrio.makeFluxLikeDA(differr, difftime, flux_units,
                                          time_units, name='err')
    diffdata['mask'] = xrio.makeFluxLikeDA(diffmask, difftime, "None",
                                           time_units, name='mask')
    variance = np.zeros_like(diffdata.flux.values)
    diffdata['variance'] = xrio.makeFluxLikeDA(variance, difftime, flux_units,
                                               time_units, name='variance')
    diffdata['guess'] = (['time'], guess)

    meta.diffmask.append(diffmask)

    return diffdata, meta, log


def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam that also works for WFC3

    Parameters
    ----------
    data : DataClass
        The data object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : DataClass
        The updated data object with outlier background pixels flagged.
    '''
    return nircam.flag_bg(data, meta)


def fit_bg(dataim, datamask, datav0, datavariance, n, meta, isplots=0):
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
    bg, mask, n = nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)

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
    data : DataClass
        The data object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The current file number.

    Returns
    -------
    data : DataClass
        The updated DataClass object.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The current log.
    """
    def writeDrift2D(arg):
        drift2D, m, n = arg
        # Assign to array of spectra and uncertainties
        meta.drift2D[-1][n] = drift2D
        return

    # Save the reference frame for each scan direction if not yet done
    if len(meta.subdata_ref) < 2:
        # FINDME: This requires that the reference files be the first
        # two files. Using other files as the reference files will
        # require loading in all of the frames at once. This will still work
        # for observations with only one scan direction, since the second ref
        # file will never be used.
        meta.subdata_ref.append(data.flux)
        meta.subdiffmask_ref.append(meta.subdiffmask[-1])

    log.writelog("  Calculating 2D drift...", mute=(not meta.verbose))
    # FINDME: instead of calculating scanHeight, consider fitting
    # stretch factor
    drift2D = np.zeros((meta.nreads, 2))
    meta.drift2D.append(drift2D)
    if meta.ncpu == 1:
        # Only 1 CPU
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p = meta.scandir[m]
        for n in range(meta.nreads):
            writeDrift2D(hst.calcDrift2D((meta.subdata_ref[p][0] *
                                          meta.subdiffmask_ref[p][0]),
                                         (data.flux[n] *
                                          data.mask[n]),
                                         m, n))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p = meta.scandir[m]
        for n in range(meta.nreads):
            res = pool.apply_async(hst.calcDrift2D,
                                   args=((meta.subdata_ref[p][0] *
                                          meta.subdiffmask_ref[p][0]),
                                         (data.flux[n] *
                                          data.mask[n]),
                                         m, n),
                                   callback=writeDrift2D)
        pool.close()
        pool.join()
        res.wait()

    log.writelog("  Performing rough, pixel-scale drift correction...",
                 mute=(not meta.verbose))
    meta.drift2D_int.append(np.round(meta.drift2D[-1], 0))
    # Correct for drift by integer pixel numbers, no interpolation
    for n in range(meta.nreads):
        data.flux[n] = spni.shift(data.flux[n],
                                  -1*meta.drift2D_int[-1][n, ::-1], order=0,
                                  mode='constant', cval=0)
        data.mask[n] = spni.shift(data.mask[n],
                                  -1*meta.drift2D_int[-1][n, ::-1], order=0,
                                  mode='constant', cval=0)
        data.variance[n] = spni.shift(data.variance[n],
                                      -1*meta.drift2D_int[-1][n, ::-1],
                                      order=0, mode='constant', cval=0)
        data.bg[n] = spni.shift(data.bg[n],
                                -1*meta.drift2D_int[-1][n, ::-1], order=0,
                                mode='constant', cval=0)

    # FINDME: The following cannot be run since we don't have the
    # full time axis.
    # Outlier rejection of full frame along time axis
    # print("Performing full-frame outlier rejection...")
    # for p in range(2):
    #     iscan   = np.where(ev.scandir == p)[0]
    #     if len(iscan) > 0:
    #         for n in range(meta.nreads):
    #             #y1  = data.guess[ev.iref,n] - meta.spec_hw
    #             #y2  = data.guess[ev.iref,n] + meta.spec_hw
    #             #estsig      = [differr[ev.iref,n,y1:y2]
    #                             for j in range(len(ev.sigthresh))]
    #             shiftmask[iscan,n] = sigrej.sigrej(shiftdata[iscan,n],
    #                                                ev.sigthresh,
    #                                                shiftmask[iscan,n])  # ,
    #                                                # estsig)

    log.writelog("  Performing sub-pixel drift correction...",
                 mute=(not meta.verbose))
    # Get indices for each pixel
    ix = range(meta.subnx)
    iy = range(meta.subny)
    # Define the degrees of the bivariate spline
    kx, ky = (1, 1)  # FINDME: should be using (3,3)
    # Correct for drift
    for n in range(meta.nreads):
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p = meta.scandir[m]
        # Need to swap ix and iy because of numpy
        spline = spi.RectBivariateSpline(iy, ix, data.flux[n], kx=kx,
                                         ky=ky, s=0)
        # Need to subtract drift2D since documentation says (where im1 is
        # the reference image)
        # "Measures the amount im2 is offset from im1 (i.e., shift im2 by
        # -1 * these #'s to match im1)"
        data.flux[n] = spline((iy-meta.drift2D[-1][n, 1] +
                               meta.drift2D_int[-1][n, 1]).flatten(),
                              (ix-meta.drift2D[-1][n, 0] +
                               meta.drift2D_int[-1][n, 0]).flatten())
        # Need to be careful with shifting the mask. Do the shifting, and
        # mask whichever pixel was closest to the one that had been masked
        spline = spi.RectBivariateSpline(iy, ix, data.mask[n], kx=kx,
                                         ky=ky, s=0)
        data.mask[n] = spline((iy-meta.drift2D[-1][n, 1] +
                               meta.drift2D_int[-1][n, 1]).flatten(),
                              (ix-meta.drift2D[-1][n, 0] +
                               meta.drift2D_int[-1][n, 0]).flatten())
        # Fractional masking won't work - make sure it is all integer
        data.mask[n] = np.round(data.mask[n]).astype(int)
        spline = spi.RectBivariateSpline(iy, ix, data.variance[n], kx=kx,
                                         ky=ky, s=0)
        data.variance[n] = spline((iy-meta.drift2D[-1][n, 1] +
                                   meta.drift2D_int[-1][n, 1]).flatten(),
                                  (ix-meta.drift2D[-1][n, 0] +
                                   meta.drift2D_int[-1][n, 0]).flatten())
        spline = spi.RectBivariateSpline(iy, ix, data.bg[n], kx=kx,
                                         ky=ky, s=0)
        data.bg[n] = spline((iy-meta.drift2D[-1][n, 1] +
                             meta.drift2D_int[-1][n, 1]).flatten(),
                            (ix-meta.drift2D[-1][n, 0] +
                             meta.drift2D_int[-1][n, 0]).flatten())

    return data, meta, log
