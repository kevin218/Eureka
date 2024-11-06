# NIRSpec specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam, sigrej
from ..lib.util import read_time, supersample


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's NIRSpec instrument.

    Parameters
    ----------
    filename : str
        Single filename to read.
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The current log.

    '''
    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    meta.filter = data.attrs['mhdr']['GRATING']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    wave_2d = hdulist['WAVELENGTH', 1].data

    try:
        data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
        data.attrs['intend'] = data.attrs['mhdr']['INTEND']
    except:
        # FINDME: Need to only catch the particular exception we expect
        log.writelog('  WARNING: Manually setting INTSTART to 0 and INTEND '
                     'to NINTS')
        data.attrs['intstart'] = 0
        data.attrs['intend'] = sci.shape[0]

    meta.photometry = False  # Photometry for NIRSpec not implemented yet.

    # Increase pixel resolution along cross-dispersion direction
    if meta.expand > 1:
        log.writelog(f'    Super-sampling y axis from {sci.shape[1]} ' +
                     f'to {sci.shape[1]*meta.expand} pixels...',
                     mute=(not meta.verbose))
        sci = supersample(sci, meta.expand, 'flux', axis=1)
        err = supersample(err, meta.expand, 'err', axis=1)
        dq = supersample(dq, meta.expand, 'cal', axis=1)
        v0 = supersample(v0, meta.expand, 'flux', axis=1)
        wave_2d = supersample(wave_2d, meta.expand, 'wave', axis=0)

    # Record integration mid-times in BMJD_TDB
    int_times = hdulist['INT_TIMES', 1].data
    if meta.time_file is not None:
        time = read_time(meta, data, log)
    elif len(int_times['int_mid_BJD_TDB']) == 0:
        # There is no time information in the simulated NIRSpec data
        if meta.firstFile:
            log.writelog('  WARNING: The timestamps for simulated MIRI data '
                         'are not in the .fits files, so using integration '
                         'number as the time value instead.')
        time = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'],
                           data.intend)
    else:
        time = int_times['int_mid_BJD_TDB']
    if len(time) > len(sci):
        # This line is needed to still handle the simulated data
        # which had the full time array for all segments
        time = time[data.attrs['intstart']:data.attrs['intend']]

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    time_units = 'BMJD_TDB'
    wave_units = 'microns'

    if (meta.firstFile and meta.spec_hw == meta.spec_hw_range[0] and
            meta.bg_hw == meta.bg_hw_range[0]):
        # Only apply super-sampling expansion once
        meta.ywindow[0] *= meta.expand
        meta.ywindow[1] *= meta.expand

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0')
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    return data, meta, log


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

    bgdata1 = data.flux[:, :meta.bg_y1]
    bgmask1 = data.mask[:, :meta.bg_y1]
    bgdata2 = data.flux[:, meta.bg_y2:]
    bgmask2 = data.mask[:, meta.bg_y2:]
    if meta.use_estsig:
        # This might not be necessary for real data
        bgerr1 = np.ma.median(np.ma.masked_equal(data.err[:, :meta.bg_y1], 0))
        bgerr2 = np.ma.median(np.ma.masked_equal(data.err[:, meta.bg_y2:], 0))
        estsig1 = [bgerr1 for j in range(len(meta.bg_thresh))]
        estsig2 = [bgerr2 for j in range(len(meta.bg_thresh))]
    else:
        estsig1 = None
        estsig2 = None
    data['mask'][:, :meta.bg_y1] = sigrej.sigrej(bgdata1, meta.bg_thresh,
                                                 bgmask1, estsig1)
    data['mask'][:, meta.bg_y2:] = sigrej.sigrej(bgdata2, meta.bg_thresh,
                                                 bgmask2, estsig2)

    return data


def flag_ff(data, meta, log):
    '''Outlier rejection of full frame along time axis.
    For data with deep transits, there is a risk of masking good transit data.
    Proceed with caution.

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
        The updated Dataset object with outlier pixels flagged.
    '''
    return nircam.flag_ff(data, meta, log)


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam which works for NIRSpec.

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        A boolean array of which data (set to True) should be masked.
    n : int
        The current integration.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
        The plotting verbosity, by default 0.

    Returns
    -------
    bg : ndarray (2D)
        The fitted background level.
    mask : ndarray (2D)
        The updated boolean mask after background subtraction, where True
        values should be masked.
    n : int
        The current integration number.
    """
    return nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Uses the code written for NIRCam which works for NIRSpec.

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
        The mask values over the aperture region. True values should be masked.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version based on the code in s3_reduce.py
    """
    return nircam.cut_aperture(data, meta, log)


def calibrated_spectra(data, meta, log, cutoff=1e-4):
    """Modify data to compute calibrated spectra in units of mJy.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    cutoff : float
        Flux values above the cutoff will be set to zero.

    Returns
    -------
    data : ndarray
        The flux values in mJy

    Notes
    -----
    History:

    - 2023-07-17, KBS
        Initial version.
    """
    # Mask uncalibrated BG region
    log.writelog("  Setting uncalibrated pixels to zero...",
                 mute=(not meta.verbose))
    boolmask = np.abs(data.flux.data) > cutoff
    data['flux'].data = np.where(boolmask, 0, data.flux.data)
    log.writelog(f"    Zeroed {np.sum(boolmask)} " +
                 "pixels in total.", mute=(not meta.verbose))

    # Convert from MJy to mJy
    log.writelog("  Converting from MJy to mJy...",
                 mute=(not meta.verbose))
    data['flux'].data *= 1e9
    data['err'].data *= 1e9
    data['v0'].data *= 1e9

    # Update units
    data['flux'].attrs["flux_units"] = 'mJy'
    data['err'].attrs["flux_units"] = 'mJy'
    data['v0'].attrs["flux_units"] = 'mJy'

    return data
