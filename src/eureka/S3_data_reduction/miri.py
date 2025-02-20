import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import background, nircam, straighten, plots_s3
try:
    from jwst import datamodels
except ImportError:
    print('WARNING: Unable to load the jwst package. As a result, the MIRI '
          'wavelength solution will not be able to be calculated in Stage 3.')
from ..lib.util import read_time, supersample

__all__ = ['read', 'straighten_trace', 'flag_ff', 'flag_bg', 'flag_bg_phot',
           'fit_bg', 'cut_aperture', 'standard_spectrum', 'clean_median_flux',
           'calibrated_spectra', 'residualBackground', 'lc_nodriftcorr']


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's MIRI instrument.

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
        The metadata object.
    log : logedit.Logedit
        The current log.

    Notes
    -----
    History:

    - Nov 2012 Kevin Stevenson
        Initial Version
    - May 2021  Kevin Stevenson
        Updated for NIRCam
    - Jun 2021  Taylor Bell
        Updated docs for MIRI
    - Jun 2021  Sebastian Zieba
        Updated for MIRI
    - Apr 2022  Sebastian Zieba
        Updated wavelength array
    - Apr 21, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''
    hdulist = fits.open(filename)

    # Load main and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data

    try:
        data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
        data.attrs['intend'] = data.attrs['mhdr']['INTEND']
    except:
        # FINDME: Need to only catch the particular exception we expect
        log.writelog('  WARNING: Manually setting INTSTART to 0 and INTEND '
                     'to NINTS')
        data.attrs['intstart'] = 0
        data.attrs['intend'] = sci.shape[0]

    if meta.photometry:
        # Working on photometry data
        meta.filter = hdulist[0].header['FILTER']

        # The DISPAXIS argument does not exist in the header of the photometry
        # data. Added it here so that code in other sections doesn't have to
        # be changed
        data.attrs['shdr']['DISPAXIS'] = 1

        # FINDME: make this better for all filters
        if meta.filter == 'F560W':
            meta.phot_wave = 5.60
        elif meta.filter == 'F770W':
            meta.phot_wave = 7.70
        elif meta.filter == 'F1000W':
            meta.phot_wave = 10.00
        elif meta.filter == 'F1130W':
            meta.phot_wave = 11.30
        elif meta.filter == 'F1280W':
            meta.phot_wave = 12.80
        elif meta.filter == 'F1500W':
            meta.phot_wave = 15.00
        elif meta.filter == 'F1800W':
            meta.phot_wave = 18.00
        elif meta.filter == 'F2100W':
            meta.phot_wave = 21.00
        elif meta.filter in ['F2550W', 'F2550WR']:
            meta.phot_wave = 25.50

        wave_1d = np.ones_like(sci[0, 0]) * meta.phot_wave
    else:
        # Working on spectroscopic data
        meta.filter = 'LRS'

        # If wavelengths are all zero or missing --> use jwst to get
        # wavelengths. Otherwise use the wavelength array from the header
        try:
            hdulist['WAVELENGTH', 1]
            wl_missing = False
        except:
            if meta.firstFile:
                log.writelog('  WAVELENGTH extension not found, using '
                             'miri.wave_MIRI_jwst function instead.')
            wl_missing = True

        if wl_missing or np.all(hdulist['WAVELENGTH', 1].data == 0):
            wave_2d = wave_MIRI_jwst(filename, meta, log)
        else:
            wave_2d = hdulist['WAVELENGTH', 1].data

        # Increase pixel resolution along cross-dispersion direction
        if meta.expand > 1:
            log.writelog(f'    Super-sampling x axis from {sci.shape[2]} ' +
                         f'to {sci.shape[2]*meta.expand} pixels...',
                         mute=(not meta.verbose))
            sci = supersample(sci, meta.expand, 'flux', axis=2)
            err = supersample(err, meta.expand, 'err', axis=2)
            dq = supersample(dq, meta.expand, 'cal', axis=2)
            v0 = supersample(v0, meta.expand, 'flux', axis=2)
            wave_2d = supersample(wave_2d, meta.expand, 'wave', axis=1)

    # Record integration mid-times in BMJD_TDB
    int_times = hdulist['INT_TIMES', 1].data
    if meta.time_file is not None:
        time = read_time(meta, data, log)
    elif len(int_times['int_mid_BJD_TDB']) == 0:
        # There is no time information in the simulated MIRI data
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

    # MIRI appears to be rotated by 90° compared to NIRCam, so rotating arrays
    # to allow the re-use of NIRCam code. Having wavelengths increase from
    # left to right on the rotated frame makes life easier
    if data.attrs['shdr']['DISPAXIS'] == 2:
        sci = np.swapaxes(sci, 1, 2)[:, :, ::-1]
        err = np.swapaxes(err, 1, 2)[:, :, ::-1]
        dq = np.swapaxes(dq, 1, 2)[:, :, ::-1]
        v0 = np.swapaxes(v0, 1, 2)[:, :, ::-1]
        wave_2d = np.swapaxes(wave_2d, 0, 1)[:, ::-1]
        if (meta.firstFile and meta.spec_hw == meta.spec_hw_range[0] and
                meta.bg_hw == meta.bg_hw_range[0]):
            # If not, we've already done this and don't want to switch it back
            if meta.ywindow[1] > 393:
                log.writelog('WARNING: The MIRI/LRS wavelength solution is '
                             'not defined for y-values > 393, while you '
                             f'have set ywindow[1] to {meta.ywindow[1]}.\n'
                             '          It is strongly recommended to set '
                             'ywindow[1] to be <= 393, otherwise you will '
                             'potentially end up with very strange results.')

            temp = np.copy(meta.ywindow)
            meta.ywindow = meta.xwindow
            meta.xwindow = sci.shape[2] - temp[::-1]
            # Only apply super-sampling expansion once
            meta.ywindow[0] *= meta.expand
            meta.ywindow[1] *= meta.expand

    if meta.photometry:
        x = None
    else:
        # Figure out the x-axis (aka the original y-axis) pixel numbers since
        # we've reversed the order of the x-axis
        x = np.arange(sci.shape[2])[::-1]

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux', x=x)
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err', x=x)
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq', x=x)
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0', x=x)
    if not meta.photometry:
        data['wave_2d'] = (['y', 'x'], wave_2d)
        data['wave_2d'].attrs['wave_units'] = wave_units
    else:
        data['wave_1d'] = (['x'], wave_1d)
        data['wave_1d'].attrs['wave_units'] = wave_units
    # Initialize bad pixel mask (False = good, True = bad)
    data['mask'] = (['time', 'y', 'x'], np.zeros(data.flux.shape, dtype=bool))

    return data, meta, log


def wave_MIRI_jwst(filename, meta, log):
    '''Compute wavelengths for simulated MIRI observations.

    This code uses the jwst package to get the wavelength information out
    of the WCS.

    Parameters
    ----------
    filename : str
        The filename of the file being read in.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    lam_x_full : list
        A list of the wavelengths

    Notes
    -----
    History:

    - August 2022  Taylor J Bell
        Initial Version
    '''
    if meta.firstFile:
        log.writelog('  WARNING: Using the jwst package because the '
                     'wavelengths are not currently in the .fits files '
                     'themselves')

    # Using the code from https://github.com/spacetelescope/jwst/pull/6964
    # as of August 8th, 2022
    with datamodels.open(filename) as model:
        data_shape = model.data.shape
        if len(data_shape) > 2:
            data_shape = data_shape[-2:]
        index_array = np.indices(data_shape[::-1])
        wcs_array = model.meta.wcs(*index_array)
        return wcs_array[2].T


def flag_bg(data, meta, log):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

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
    return nircam.flag_bg(data, meta, log)


def flag_bg_phot(data, meta, log):
    '''Outlier rejection of sky background along time axis for photometry.

    Uses the code written for NIRCam which also works for MIRI.

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
    return nircam.flag_bg_phot(data, meta, log)


def flag_ff(data, meta, log):
    '''Outlier rejection of full frame along time axis.
    For data with deep transits, there is a risk of masking good transit data.
    Proceed with caution.

    Uses the code written for NIRCam which also works for MIRI.

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

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

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
    bg, mask = background.fitbg(dataim, meta, datamask, meta.bg_y1,
                                meta.bg_y2, deg=meta.bg_deg,
                                threshold=meta.p3thresh, isrotate=2,
                                isplots=isplots)
    return bg, mask, n


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

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
    apmedflux : ndarray
        The median flux over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version based on the code in s3_reduce.py
    """
    return nircam.cut_aperture(data, meta, log)


def standard_spectrum(data, meta, apdata, apmask, aperr):
    """Instrument wrapper for computing the standard box spectrum.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    apdata : ndarray
        The pixel values in the aperture region.
    apmask : ndarray
        The outlier mask in the aperture region. True where pixels should be
        masked.
    aperr : ndarray
        The noise values in the aperture region.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object in which the spectrum data will stored.
    """

    return nircam.standard_spectrum(data, meta, apdata, apmask, aperr)


def calibrated_spectra(data, meta, log):
    """Modify data to compute calibrated spectra in units of mJy.

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
    data : ndarray
        The flux values in mJy
    """
    # Convert from MJy/sr to mJy
    log.writelog("  Converting from MJy/sr to mJy...",
                 mute=(not meta.verbose))
    data['flux'].data *= 1e9*data.shdr['PIXAR_SR']
    data['err'].data *= 1e9*data.shdr['PIXAR_SR']
    data['v0'].data *= 1e9*data.shdr['PIXAR_SR']

    # Update units
    data['flux'].attrs["flux_units"] = 'mJy'
    data['err'].attrs["flux_units"] = 'mJy'
    data['v0'].attrs["flux_units"] = 'mJy'
    return data


def straighten_trace(data, meta, log, m):
    """Instrument wrapper for computing the standard box spectrum.

    Parameters
    ----------
    data : Xarray Dataset
            The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    m : int
        The file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    return straighten.straighten_trace(data, meta, log, m)


def clean_median_flux(data, meta, log, m):
    """Instrument wrapper for computing a median flux frame that is
    free of bad pixels.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    """

    return nircam.clean_median_flux(data, meta, log, m)


def residualBackground(data, meta, m, vmin=None, vmax=None):
    """Plot the median, BG-subtracted frame to study the residual BG region and
    aperture/BG sizes. (Fig 3304)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    vmin : int; optional
        Minimum value of colormap. Default is None.
    vmax : int; optional
        Maximum value of colormap. Default is None.
    """
    plots_s3.residualBackground(data, meta, m, vmin=None, vmax=None)


def lc_nodriftcorr(spec, meta):
    '''Plot a 2D light curve without drift correction. (Fig 3101+3102)

    Fig 3101 uses a linear wavelength x-axis, while Fig 3102 uses a linear
    detector pixel x-axis.

    Parameters
    ----------
    spec : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    mad = meta.mad_s3[0]
    plots_s3.lc_nodriftcorr(meta, spec.wave_1d, spec.optspec,
                            optmask=spec.optmask, mad=mad)
