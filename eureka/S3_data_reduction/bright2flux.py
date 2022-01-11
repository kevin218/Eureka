import numpy as np
import scipy.interpolate as spi
from scipy.constants import arcsec
from astropy.io import fits

def dn2electrons(data, meta):
    """This function converts the data, uncertainty, and variance arrays from raw units (DN) to electrons.

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of DN.

    meta:   MetaClass
        The metadata object.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of electrons.

    Notes
    -----
    The gain files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    History:
    
    - Jun 2021 Kevin Stevenson
        Initial version
    - Jul 2021
        Added gainfile rotation
    """
    # Subarray parameters
    xstart  = data.mhdr['SUBSTRT1']
    ystart  = data.mhdr['SUBSTRT2']
    nx      = data.mhdr['SUBSIZE1']
    ny      = data.mhdr['SUBSIZE2']

    # Load gain array in units of e-/ADU
    gain    = fits.getdata(meta.gainfile)[ystart:ystart+ny,xstart:xstart+nx]

    # Like in the case of MIRI data, the gain file data has to be rotated by 90 degrees
    if data.shdr['DISPAXIS']==2:
        gain = np.swapaxes(gain, 0, 1)

    # Gain subarray
    subgain = gain[meta.ywindow[0]:meta.ywindow[1],meta.xwindow[0]:meta.xwindow[1]]

    # Convert to electrons
    data.subdata *= subgain
    data.suberr  *= subgain
    data.subv0   *= (subgain)**2

    return data

def bright2dn(data, meta):
    """This function converts the data, uncertainty, and variance arrays from brightness units (MJy/sr) to raw units (DN).

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr.

    meta:   MetaClass
        The metadata object.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of DN.

    Notes
    -----
    The photometry files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    History:

    - 2021-05-28 kbs
        Initial version
    - 2021-07-21 sz
        Added functionality for MIRI
    """
    # Load response function and wavelength
    foo = fits.getdata(meta.photfile)
    if meta.inst == 'nircam':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['pupil'] == data.mhdr['PUPIL']) * (foo['order'] == 1))[0][0]
    if meta.inst == 'miri':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['subarray'] == data.mhdr['SUBARRAY']))[0][0]

    response_wave = foo['wavelength'][ind]
    response_vals = foo['relresponse'][ind]
    igood = np.where(response_wave > 0)[0]
    response_wave = response_wave[igood]
    response_vals = response_vals[igood]
    # Interpolate response at desired wavelengths
    f = spi.interp1d(response_wave, response_vals, 'cubic')
    response = f(data.subwave)

    scalar = data.shdr['PHOTMJSR']
    # Convert to DN/sec
    data.subdata /= scalar * response
    data.suberr  /= scalar * response
    data.subv0   /= (scalar * response)**2
    # From DN/sec to DN
    int_time = data.mhdr['EFFINTTM']
    data.subdata *= int_time
    data.suberr  *= int_time
    data.subv0   *= int_time

    return data

def bright2flux(data, err, v0, pixel_area):
    """This function converts the data and uncertainty arrays from brightness units (MJy/sr) to flux units (Jy/pix).

    Parameters
    ----------
    data:   ndarray
            data array of shape ([nx, ny, nimpos, npos]) in units of MJy/sr.
    err:    ndarray
            uncertainties of data (same shape and units).
    v0:     ndarray
            variance array for data (same shape and units).
    pixel_area:  ndarray
            Pixel area (arcsec/pix)

    Returns
    -------
    data:   ndarray
            data array of shape ([nx, ny, nimpos, npos]) in units of Jy/pix.
    err:    ndarray
            uncertainties of data (same shape and units).
    v0:     ndarray
            variance array for data (same shape and units).

    Notes
    -----
    The input arrays Data and Uncd are changed in place.

    History:

    - 2005-06-20 Statia Luszcz, Cornell (shl35@cornell.edu).
    - 2005-10-13 jh
        Renamed, modified doc, removed posmed, fixed
        nimpos default bug (was float rather than int).
    - 2005-10-28 jh        
        Updated header to give units being converted
        from/to, made srperas value a calculation
        rather than a constant, added Allen reference.
    - 2005-11-24 jh
        Eliminated NIMPOS.
    - 2008-06-28 jh
        Allow npos=1 case.
    - 2010-01-29 patricio (pcubillos@fulbrightmail.org)
        Converted to python. 
    - 2010-11-01 patricio
        Documented, and incorporated scipy.constants.
    - 2021-05-28 kbs
        Updated for JWST
    """
    # steradians per square arcsecond
    srperas = arcsec**2.0

    data *= srperas * 1e6 * pixel_area
    err  *= srperas * 1e6 * pixel_area
    v0   *= srperas * 1e6 * pixel_area

    return data, err, v0

def convert_to_e(data, meta, log):
    """This function converts the data object to electrons from MJy/sr or DN/s.

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr or DN/s.

    meta:   MetaClass
        The metadata object.

    log:    logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of electrons.

    meta:   MetaClass
        The metadata object.
    """
    if data.shdr['BUNIT'] == 'MJy/sr':
        # Convert from brightness units (MJy/sr) to flux units (uJy/pix)
        # log.writelog('Converting from brightness to flux units')
        # subdata, suberr, subv0 = b2f.bright2flux(subdata, suberr, subv0, shdr['PIXAR_A2'])
        # Convert from brightness units (MJy/sr) to DNs
        log.writelog('  Converting from brightness units (MJy/sr) to electrons')
        meta.photfile = meta.topdir + meta.ancildir + '/' + data.mhdr['R_PHOTOM'][7:]
        data = bright2dn(data, meta)
        meta.gainfile = meta.topdir + meta.ancildir + '/' + data.mhdr['R_GAIN'][7:]
        data = dn2electrons(data, meta)
    if data.shdr['BUNIT'] == 'DN/s':
        # Convert from DN/s to e/s
        log.writelog('  Converting from data numbers per second (DN/s) to electrons per second (e/s)')
        meta.gainfile = meta.topdir + meta.ancildir + '/' + data.mhdr['R_GAIN'][7:]
        data = dn2electrons(data, meta)
    return data, meta
