import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
try:
    from jwst import datamodels
except ImportError:
    print('WARNING: Unable to load the jwst package. As a result, the MIRI '
          'wavelength solution will not be able to be calculated in Stage 3.')
from . import nircam
from ..lib.util import read_time


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
    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
    data.attrs['intend'] = data.attrs['mhdr']['INTEND']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data

    meta.photometry = False  # Photometry for MIRI not implemented yet.

    # If wavelengths are all zero --> use jwst to get wavelengths
    # Otherwise use the wavelength array from the header
    if np.all(hdulist['WAVELENGTH', 1].data == 0):
        wave_2d = wave_MIRI_jwst(filename, meta, log)
    else:
        wave_2d = hdulist['WAVELENGTH', 1].data
    int_times = hdulist['INT_TIMES', 1].data

    # Record integration mid-times in BMJD_TDB
    if (hasattr(meta, 'time_file') and meta.time_file is not None):
        time = read_time(meta, data, log)
    elif len(int_times['int_mid_BJD_TDB']) == 0:
        if meta.firstFile:
            log.writelog('  WARNING: The timestamps for the simulated MIRI '
                         'data are currently hardcoded because they are not '
                         'in the .fits files themselves')
        if ('WASP_80b' in data.attrs['filename']
                and 'transit' in data.attrs['filename']):
            # Time array for WASP-80b MIRISIM transit observations
            # Assuming transit near August 1, 2022
            phase_i = 0.95434
            phase_f = 1.032726
            t0 = 2456487.425006
            per = 3.06785234
            time_i = phase_i*per+t0
            while np.abs(time_i-2459792.54237) > per:
                time_i += per
            time_f = phase_f*per+t0
            while time_f < time_i:
                time_f += per
            time = np.linspace(time_i, time_f, 4507,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif ('WASP_80b' in data.attrs['filename']
              and 'eclipse' in data.attrs['filename']):
            # Time array for WASP-80b MIRISIM eclipse observations
            # Assuming eclipse near August 1, 2022
            phase_i = 0.45434
            phase_f = 0.532725929856498
            t0 = 2456487.425006
            per = 3.06785234
            time_i = phase_i*per+t0
            while np.abs(time_i-2459792.54237) > per:
                time_i += per
            time_f = phase_f*per+t0
            while time_f < time_i:
                time_f += per
            time = np.linspace(time_i, time_f, 4506,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif 'new_drift' in data.attrs['filename']:
            # Time array for the newest MIRISIM observations
            time = np.linspace(0, 47.712*(1849)/3600/24, 1849,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif data.attrs['mhdr']['EFFINTTM'] == 10.3376:
            # There is no time information in the old simulated MIRI data
            # As a placeholder, I am creating timestamps indentical to the
            # ones in STSci-SimDataJWST/MIRI/Ancillary_files/times.dat.txt
            # converted to days
            time = np.linspace(0, 17356.28742796742/3600/24, 1680,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']]
        elif data.attrs['mhdr']['EFFINTTM'] == 47.712:
            # A new manually created time array for the new MIRI simulations
            # Need to subtract an extra 1 from intend for these data
            time = np.linspace(0, 47.712*(42*44-1)/3600/24, 42*44,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        else:
            raise AssertionError('Eureka does not currently know how to '
                                 'generate the time array for these'
                                 'simulations.')
    else:
        time = int_times['int_mid_BJD_TDB']

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
            temp = np.copy(meta.ywindow)
            meta.ywindow = meta.xwindow
            meta.xwindow = sci.shape[2] - temp[::-1]
    
    # Figure out the x-axis (aka the original y-axis) pixel numbers since we've
    # reversed the order of the x-axis
    x = np.arange(sci.shape[2])[::-1]

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux', x=x)
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err', x=x)
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq', x=x)
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0', x=x)
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

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
    # as of August 8th
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


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        An array of which data should be masked.
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
        The updated mask after background subtraction.
    n : int
        The current integration number.
    """
    return nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)


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
        The mask values over the aperture region.
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
