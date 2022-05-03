# NIRSpec specific rountines go here
import numpy as np
from astropy.io import fits
from . import nircam, sigrej


def read(filename, data, meta):
    '''Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename : str
        Single filename to read.
    data : DataClass
        The data object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : DataClass
        The updated data object with the fits data stored inside.

    Notes
    -----
    History:

    - November 2012 Kevin Stevenson
        Initial version
    - June 2021 Aarynn Carter/Eva-Maria Ahrer
        Updated for NIRSpec
    '''
    # Now we can start working with the data.
    hdulist = fits.open(filename)
    data.filename = filename
    data.mhdr = hdulist[0].header
    data.shdr = hdulist['SCI', 1].header

    data.intstart = 1

    try:
        data.intstart = data.mhdr['INTSTART']
        data.intend = data.mhdr['INTEND']
    except:
        # FINDME: Need to only catch the particular exception we expect
        print('  WARNING: Manually setting INTSTART to 1 and INTEND to NINTS')
        data.intstart = 1
        data.intend = data.mhdr['NINTS']

    data.data = hdulist['SCI', 1].data
    data.err = hdulist['ERR', 1].data
    data.dq = hdulist['DQ', 1].data
    data.wave = hdulist['WAVELENGTH', 1].data
    data.v0 = hdulist['VAR_RNOISE', 1].data
    int_times = hdulist['INT_TIMES', 1].data[data.intstart-1:data.intend]

    # Record integration mid-times in BJD_TDB
    if len(int_times['int_mid_BJD_TDB']) == 0:
        # There is no time information in the simulated NIRSpec data
        print('  WARNING: The timestamps for the simulated NIRSpec data are '
              '           currently hardcoded because they are not in the '
              '           .fits files themselves')
        data.time = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'],
                                data.intend)
    else:
        data.time = int_times['int_mid_BJD_TDB']
    meta.time_units = 'BJD_TDB'

    return data, meta


def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

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
    y1, y2, bg_thresh = meta.bg_y1, meta.bg_y2, meta.bg_thresh

    bgdata1 = data.subdata[:, :y1]
    bgmask1 = data.submask[:, :y1]
    bgdata2 = data.subdata[:, y2:]
    bgmask2 = data.submask[:, y2:]
    # This might not be necessary for real data
    bgerr1 = np.ma.median(np.ma.masked_equal(data.suberr[:, :y1], 0))
    bgerr2 = np.ma.median(np.ma.masked_equal(data.suberr[:, y2:], 0))

    estsig1 = [bgerr1 for j in range(len(bg_thresh))]
    estsig2 = [bgerr2 for j in range(len(bg_thresh))]

    data.submask[:, :y1] = sigrej.sigrej(bgdata1, bg_thresh, bgmask1, estsig1)
    data.submask[:, y2:] = sigrej.sigrej(bgdata2, bg_thresh, bgmask2, estsig2)

    return data


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam which works for NIRSpec.

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
    isplots : int, optional
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
